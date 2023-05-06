import datetime
from difflib import SequenceMatcher
import os

from pathlib import Path
from datasets import load_dataset
from bertalign import Bertalign
import process_zh_text
import process_en_text
import numpy as np
import pickle
import json
from collections import OrderedDict, Counter
import re
from datasets.dataset_dict import DatasetDict
##
from sentence_transformers import SentenceTransformer

from helper import filter_leading_and_tail_blank_lines, dump_row, make_filter_log
from helper import PAGINATION_TOKEN, LANGS



HEADER_SCAN_LIMIT = 100
DIGITS_PATTERN = re.compile('^\d+$')
MATCHER_RATIO = 0.72

def drop_pagination_header_and_footer(row: DatasetDict):
    """语种无关过滤页眉（包括页码），不依赖任何正则，仅依靠自身和其它语种中出现的文本块频度统计实现
    
    Args：
        row (DatasetDict): datasets map进来的行，内含一篇文章的六个语种版本，每页用\n----\n隔开
    Returns:
        row (DatasetDict): 清洗后按原格式组装的row
    """
    record = row['record']

    file_content = {}
    token_spots = {}
    line_spots = {}
    page_token_slotses = {}

    overall_token_spots = Counter()
    overall_pages_num = 0

    # maxpage = 0
    for lang in LANGS:
        file_content[lang] = pages = row[lang].split(PAGINATION_TOKEN)
        overall_pages_num += len(pages)
    #     maxpage = max(maxpage, len(pages))
        token_spots[lang] = token_spot = Counter() # token计数表，只用来完全匹配，其中页码特判
        line_spots[lang] = line_spot = Counter() # 行计数表，比token粒度更大，用于difflib的近似匹配
        page_token_slotses[lang] = page_token_slots = [set() for _ in pages] # 每个用来装页眉的token，仅用来判断疑似页码的数字

        for pageid, page in enumerate(pages):
            lines = page.strip().split('\n')
            page = pages[pageid] = '\n'.join(lines)

            for lineid, line in enumerate(page[:HEADER_SCAN_LIMIT].split('\n')): # 页眉只最多取前100字符
                for token in line.split(' '):
                    # if len(token) < 2: # 单字符的token太危险，不能要
                        # continue
                    page_token_slots[pageid].add(token)
                    token_spot[token] += 1
                line_digest = line.replace(' ', '')
                if line_digest:
                    line_spot[line_digest] += 1 # 行计数表是用于尝试清除类似P a g e 2这种形式的页码
        
        for token, ctr in token_spot.items():
            overall_token_spots[token] += ctr

        # 去掉只出现少数的token，提高效率
        for x in list(token_spot.keys()):
            if token_spot[x] < 3 or token_spot[x] > len(pages):
                token_spot.pop(x)

    for lang, pages in file_content.items():
        token_spot = token_spots[lang]
        line_spot = line_spots[lang]
        page_token_slots = page_token_slotses[lang]

        pagination_offset = 1
        maxcombo = 0
        for offset in range(-9, 3): # 0 1 2
            combo = 0
            for pageid in range(len(pages)):
                if str(pageid + offset) in page_token_slots[pageid]:
                    combo += 1
            if combo > maxcombo:
                maxcombo = combo
                pagination_offset = offset
        # if maxcombo < len(pages) // 2:
        #     pagination_offset = None

        def is_freq(freq: int): return len(pages) >= 3 and freq >= len(pages) - 1 or len(pages) >= 5 and freq > len(pages) * 2 / 3

        for pageid, page in enumerate(pages):
            header, body = page[:HEADER_SCAN_LIMIT], page[HEADER_SCAN_LIMIT:]
            newlines = []
            done = False # 我们只删连续一段开头的，这样写来防止删掉类似the la de这些常见单词

            for line in header.split('\n'):
                # if 'A/CN.9/WG.VI/WP.22/Add.1' in line and lang == 'zh':
                #     print('break')
                # else:
                #     continue
                # # 行近似匹配
                line = line.strip()
                if not line or done: # 空行不管，先照旧插入newlines
                    newlines.append(line)
                    continue

                line_digest = line.replace(' ', '')
                
                # substr_score = Counter() # LCS得分，用于处理最长公共子序列，情况不多且过于复杂，先不用，这里留个想法
                line_freq = 0
                for line_str, ctr in line_spot.items():
                    matcher = SequenceMatcher(None, line_digest, line_str, autojunk=False)
                    # 上界剪枝
                    if matcher.real_quick_ratio() > MATCHER_RATIO and matcher.quick_ratio() > MATCHER_RATIO and matcher.ratio() > MATCHER_RATIO:
                        line_freq += ctr
                if is_freq(line_freq):
                    make_filter_log(line, record, lang, pageid, f'line_freq: {line_freq}, pages: {len(pages)}')
                    continue
                # 按token过滤
                new_tokens = []
                for token in line.split(' '):
                    # token.isdigit() 不可靠
                    if not token: continue
                    if not done:
                        if pagination_offset is not None and re.match(DIGITS_PATTERN, token) and int(token) == pageid + pagination_offset: # 特判页码
                            make_filter_log(token, record, lang, pageid, f'likely page number')
                            continue
                        overall_token_freq = overall_token_spots[token]
                        if overall_token_freq > overall_pages_num // 2:
                            make_filter_log(token, record, lang, pageid, f'overall_tk_freq: {overall_token_freq}, all_pages: {overall_pages_num}')
                            continue
                        # for token_str, ctr in token_spot.items():
                            # if token_str == token:
                                # token_freq += ctr
                        token_freq = token_spot[token]
                        if is_freq(token_freq) and not token_freq > len(pages):
                            make_filter_log(token, record, lang, pageid, f'tk_freq: {token_freq}, pages: {len(pages)}')
                            continue

                    new_tokens.append(token)
                    done = True

                newlines.append(' '.join(new_tokens))
            
            # 去页脚逻辑
            annotation_index = body.find('__________')
            if annotation_index != -1:
                make_filter_log(body[annotation_index:], record, lang, pageid, f"annotation block")
                body = body[:annotation_index]

            pages[pageid] = ('\n'.join(newlines) + body).strip()
        row[lang] = PAGINATION_TOKEN.join(pages) # 放回row，统一格式，之后用别的函数处理合页与成段

import process_en_text
import process_zh_text
from helper import dump_align_result_to_file


def debug(row: DatasetDict):
    drop_pagination_header_and_footer(row)
    for lang in LANGS:
        if lang != 'zh':
            row[lang] = process_en_text.start(row[lang])
        else:
            row[lang] = process_zh_text.start(row[lang])
    dump_row(row)
    ba = Bertalign(row)
    ba.align_sents()
    result = ba.create_result()
    dump_align_result_to_file(row['record'], result)

def debug_init():
    for lang in LANGS:
        try:
            os.remove(r'C:\Users\Administrator\Documents\parallel_corpus_mnbvc\alignment\bertalign\pre' + f'/dbg_{lang}.txt')
        except Exception as e:
            print(e)
    
    try:
        os.remove(r'C:\Users\Administrator\Documents\parallel_corpus_mnbvc\alignment\bertalign\filter_log.jsonl')
    except Exception as e:
        print(e)





if __name__ == "__main__":
    # os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    begin_time = datetime.datetime.now()
    debug_init()
    # dataset = load_dataset("ranWang/UN_Historical_PDF_Article_Text_Corpus", split='randomTest')
    dataset = load_dataset("ranWang/UN_PDF_TEXT_DATA_TEST", split='randomTest')
    dataset.map(debug)
    # print(len(VECTORS))
    # make_marked_file()
    # visualize()
    end_time = datetime.datetime.now()
    print('Time elapsed:', end_time - begin_time)
