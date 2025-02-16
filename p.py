import datetime
from difflib import SequenceMatcher
import itertools
import os

from pathlib import Path
import traceback
from datasets import load_dataset, load_from_disk
from bertalign import Bertalign
import process_zh_text
import process_en_text
import numpy as np
import json
from collections import OrderedDict, Counter
import re
from datasets.dataset_dict import DatasetDict
##
from sentence_transformers import SentenceTransformer

from helper import dump_row, ensure_dirs, make_filter_log, my_path, align_logger, read_secret, use_proxy
from helper import PAGINATION_TOKEN, LANGS
from helper import dump_align_result_to_file
from helper import PREPROCESS_DIR, ALIGNED_DIR, FILTER_LOG, ERROR_LOG

# 禁用log
make_filter_log = lambda *x: None

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
        token_spots[lang] = token_spot = Counter()  # token计数表，只用来完全匹配，其中页码特判
        # 行计数表，比token粒度更大，用于difflib的近似匹配
        line_spots[lang] = line_spot = Counter()
        page_token_slotses[lang] = page_token_slots = [
            set() for _ in pages]  # 每个用来装页眉的token，仅用来判断疑似页码的数字

        for pageid, page in enumerate(pages):
            lines = page.strip().splitlines()
            page = pages[pageid] = '\n'.join(lines)

            # 页眉只最多取前100字符
            for lineid, line in enumerate(page[:HEADER_SCAN_LIMIT].splitlines()):
                for token in line.split():
                    # if len(token) < 2: # 单字符的token太危险，不能要
                    # continue
                    page_token_slots[pageid].add(token)
                    token_spot[token] += 1
                line_digest = line.replace(' ', '')
                if line_digest:
                    # 行计数表是用于尝试清除类似P a g e 2这种形式的页码
                    line_spot[line_digest] += 1

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
        for offset in range(-9, 3):  # 0 1 2
            combo = 0
            for pageid in range(len(pages)):
                if str(pageid + offset) in page_token_slots[pageid]:
                    combo += 1
            if combo > maxcombo:
                maxcombo = combo
                pagination_offset = offset
        # if maxcombo < len(pages) // 2:
        #     pagination_offset = None

        def is_freq(freq: int): return len(pages) >= 3 and freq >= len(
            pages) - 1 or len(pages) >= 5 and freq > len(pages) * 2 / 3

        for pageid, page in enumerate(pages):
            header, body = page[:HEADER_SCAN_LIMIT], page[HEADER_SCAN_LIMIT:]
            newlines = []
            done = False  # 我们只删连续一段开头的，这样写来防止删掉类似the la de这些常见单词

            for line in header.splitlines():
                # if 'A/CN.9/WG.VI/WP.22/Add.1' in line and lang == 'zh':
                #     print('break')
                # else:
                #     continue
                # # 行近似匹配
                line = line.strip()
                if not line or done:  # 空行不管，先照旧插入newlines
                    # if line: # 这行用来管空行，即丢掉空行
                    newlines.append(line)
                    continue

                line_digest = line.replace(' ', '')

                # substr_score = Counter() # LCS得分，用于处理最长公共子序列，情况不多且过于复杂，先不用，这里留个想法
                line_freq = 0
                for line_str, ctr in line_spot.items():
                    matcher = SequenceMatcher(
                        None, line_digest, line_str, autojunk=False)
                    # 上界剪枝
                    if matcher.real_quick_ratio() > MATCHER_RATIO and matcher.quick_ratio() > MATCHER_RATIO and matcher.ratio() > MATCHER_RATIO:
                        line_freq += ctr
                if is_freq(line_freq):
                    make_filter_log(line, record, lang, pageid,
                                    f'line_freq: {line_freq}, pages: {len(pages)}')
                    continue
                # 按token过滤
                new_tokens = []
                for token in line.split(' '):
                    # token.isdigit() 不可靠
                    if not token:
                        continue
                    if not done:
                        # 特判页码
                        if pagination_offset is not None and re.match(DIGITS_PATTERN, token) and int(token) == pageid + pagination_offset:
                            make_filter_log(token, record, lang,
                                            pageid, f'likely page number')
                            continue
                        overall_token_freq = overall_token_spots[token]
                        if overall_token_freq > overall_pages_num // 2:
                            make_filter_log(
                                token, record, lang, pageid, f'overall_tk_freq: {overall_token_freq}, all_pages: {overall_pages_num}')
                            continue
                        # for token_str, ctr in token_spot.items():
                            # if token_str == token:
                            # token_freq += ctr
                        token_freq = token_spot[token]
                        if is_freq(token_freq) and not token_freq > len(pages):
                            make_filter_log(
                                token, record, lang, pageid, f'tk_freq: {token_freq}, pages: {len(pages)}')
                            continue

                    new_tokens.append(token)
                    done = True

                newlines.append(' '.join(new_tokens))

            # 去页脚逻辑
            annotation_index = body.find('__________')
            if annotation_index != -1:
                make_filter_log(body[annotation_index:],
                                record, lang, pageid, f"annotation block")
                body = body[:annotation_index]

            pages[pageid] = ('\n'.join(newlines) + body).strip()
        row[lang] = PAGINATION_TOKEN.join(pages)  # 放回row，统一格式，之后用别的函数处理合页与成段

def remove_duplicate_breakline(pages: list[str]):
    flatten = list(line.strip() for line in itertools.chain(*[page.splitlines() for page in pages]))
    outputs = []
    for i in flatten:
        if not i:
            continue
        outputs.append(i)
    return '\n'.join(outputs)

# flg = False
def debug(row: DatasetDict):
    # global flg
    # if row['record'] == '581574':
    #     flg = True
    # if flg == False:
    #     return
    # if row['record'] != '432027':
        # return
    rec = row['record']
    drop_pagination_header_and_footer(row)
    for lang in LANGS:
        row[lang] = remove_duplicate_breakline(row[lang].split(PAGINATION_TOKEN))
        if lang != 'zh':
            row[lang] = process_en_text.start(row[lang])
        else:
            row[lang] = process_zh_text.start(row[lang])
    # row[''] = {
        
    # }
    return row

# def dump_mapping(row):
    # dump_row(row)

    # try:
    #     ba = Bertalign(row, is_splited=True, log_func=align_logger)
    #     ba.align_sents()
    #     result = ba.create_result()
    #     dump_align_result_to_file(row['record'], result)
    # except:
    #     with open(my_path(ERROR_LOG), 'a', encoding='utf-8') as f:
    #         json.dump({'time': str(datetime.datetime.now()),
    #                   'record': rec, 'err': traceback.format_exc()}, f)
    #         f.write('\n')


def debug_init():
    ensure_dirs()
    for lang in LANGS:
        try:
            os.remove(my_path(PREPROCESS_DIR, f'dbg_{lang}.txt'))
        except Exception as e:
            print(e)

    try:
        os.remove(my_path(FILTER_LOG))
    except Exception as e:
        print(e)

    try:
        os.remove(my_path(ERROR_LOG))
    except Exception as e:
        print(e)

    for f in os.listdir(my_path(ALIGNED_DIR)):
        if f.endswith('.txt'):
            os.remove(my_path(ALIGNED_DIR, f))


if __name__ == "__main__":
    # os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    begin_time = datetime.datetime.now()
    # from helper import hacked_push_to_hub
    debug_init()
    dataset = load_dataset("ranWang/UN_PDF_TEXT_DATA_TEST", split='randomTest')
    dataset = dataset.map(debug, num_proc=8)

    # from process_en_text import extract_sentences_from_single_file
    # text = 
    # extract_sentences_from_single_file(text.split(PAGINATION_TOKEN))

    # debug(dataset['en'][2000])
    dataset.map(dump_row)
    # print(dataset)
    dataset.save_to_disk(my_path())

    use_proxy()
    # dataset = load_from_disk(my_path())

    # proxy_url = 'http://localhost:7890'
    # os.environ["HTTP_PROXY"] = proxy_url
    # os.environ["HTTPS_PROXY"] = proxy_url
    hftoken = read_secret('hf_token')
    # dataset.push_to_hub('bot-yaya/UN_PDF_SUBSET_FORM_BY_RULESET', token=hftoken, max_shard_size='32MB')
    dataset.push_to_hub('bot-yaya/UN_PDF_SUBSET_PREPROCESSED', token=hftoken, max_shard_size='32MB')
    # with open(my_path('data', 'README.md'), 'wb') as rm:
        # rm.write(readme)

    # sh = dataset.shard(1, 0, contiguous=True)
    # num_shards = 1
    # def path_in_repo(_index, shard):
    #     return f"data/randomTest-{_index:05d}-of-{num_shards:05d}-{shard._fingerprint}.parquet"

    # with open(my_path(path_in_repo(0, sh)), 'wb') as fsh:
    #     sh.to_parquet(fsh)

    # print(len(VECTORS))
    # make_marked_file()
    # visualize()
    end_time = datetime.datetime.now()
    print('Time elapsed:', end_time - begin_time)



"""
I want you to split the following text paragragh by paragragh, 
but please join the adjacent lines if it can form a meaningful paragragh and left the breakline if it can split the paragragh,

I need your help to solve a breakline elimination problem,
given a text exported from PDF, 
some breakline may separate a meaningful paragragh unexpectly,
in this case, you should join adjacent lines if they can form a meaningful paragraph and replace the breakline symbol as space.
try to filter noises and keep as many meaningful info as you can, 
leave the indexing information as it is, 
do not add more word to the source input text, 
format the resulting paragraphs as python list.

are you ready for this task?


my input can contain multiple batches,
so some of the trailing lines may come from the next paragraph, in this case, 
you should memorized them and join them to my next input batch.

"""