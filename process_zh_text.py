from typing import Dict
from helper import LINEDOT_TOKEN, LINENO_TOKEN
from helper import cat_by_lineno
from helper import PAGINATION_TOKEN
import re
import datetime
import itertools
from collections import Counter, deque
import string
import jieba

import nltk
import Levenshtein
from datasets import load_dataset

# GLOBAL CONSTANTS
INDEX_TOKEN = '...'

PUNCTUATION_SET = set(string.punctuation)
PUNCTUATION_LANG = {
    'ar': {
        '،': '.',  # full stop
        '.': '.',  # full stop
        '!': '!',  # exclamation mark
        '؟': '?',  # question mark
        '،': ',',  # comma
        '؛': ';',  # semicolon
        ':': ':',  # colon
        '“': '"',  # left quotation marks
        '”': '"',  # right quotation marks
    },
    'zh': {
        '，': ',',
        '。': '.',
        '：': ':',
        '？': '?',
        '！': '!',
        '；': ';',
        '“': '"',
        '”': '"',
        '（': '(',
        '）': ')',
    },
}
for k, v in PUNCTUATION_LANG.items():
    PUNCTUATION_SET.update(v.keys())

DIGITS = {
    'ar': {
        '٠': 0,
        '١': 1,
        '٢': 2,
        '٣': 3,
        '٤': 4,
        '٥': 5,
        '٦': 6,
        '٧': 7,
        '٨': 8,
        '٩': 9,
    },
    'zh': {
        '零': 0,
        '一': 1,
        '二': 2,
        '三': 3,
        '四': 4,
        '五': 5,
        '六': 6,
        '七': 7,
        '八': 8,
        '九': 9,
        '十': 10,
    }
}

IS_ALL_THIS_LANG = {
    # \u0621-\u064A\u0660-\u0669
    # 除中文外，句子中都含空格
    'ar': re.compile(r'[\u0600-\u06ff ]+'),
    'zh': re.compile(r'[\u3006\u3007\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U0003134f]+'),
    'fr': re.compile(r'[a-zA-ZÀ-Ÿ ]+'),
    'es': re.compile(r'[a-zA-ZáéíóúñÁÉÍÓÚÑüÜ ]+'),
    'ru': re.compile(r'[А-я,Ё,ё ]+'),
    'en': re.compile(r'[A-Za-z ]+'),
}


zh_no_concat_ruleset = [
    re.compile(r'摘要$'),
    re.compile(r'注$'),
    re.compile(r'导言$'),
    re.compile(r'^附件[一二三四五六七八九十].$'),
]

zh_char = re.compile(
    r'[\u3006\u3007\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U0003134f]')


def zh_rate(src: str) -> float:
    return len(re.findall(zh_char, src)) / len(src) if len(src) else 0


def zh_is_end_punctuation(s1: str) -> bool:
    """根据标点判断能不能合并，确保s1非空"""
    # s1 = s1.strip()
    if s1[-1] in ('。', '！', '？', '!', '?', '…', '；', ';'):
        return True
    return False

def zh_isnot_end_punctuation(s1: str) -> bool:
    return s1[-1] in ('，', '、', '（', ',', '(', '"', '“', '《')

def can_concat_two_by_ruleset(s1: str, s2: str) -> bool:
    """清除中文之间的空格的规则"""
    if (r2 := zh_rate(s2)) <= 0.01 or (r1 := zh_rate(s1)) <= 0.01:  # 几乎不含中文字的，我们不合并
        return False

    back_char = s1[-1]
    front_char = s2[0]
    if zh_is_end_punctuation(back_char):  # 由标点符号能够断开的，我们不合并
        return False
    # if back_char == '。': # 特判标点符号
    #     return False
    # elif back_char in ('，', '）', '、'):
    #     return True

    match_no_concat_ruleset = False
    for pat in zh_no_concat_ruleset: # 长得像目录的，我们不动
        if re.search(pat, s2) or re.search(pat, s1):
            match_no_concat_ruleset = True
            break
    if match_no_concat_ruleset:
        return False

    # for pat in (LINEDOT_TOKEN, LINENO_TOKEN):
    #     if re.search(pat, s2):
    #         match_no_concat_ruleset = True # 与数字标号规则冲突的，我们不合并
    #         break

    # if match_no_concat_ruleset:
    #     return False

    conn = back_char + front_char
    result = jieba.cut(s1[-100:] + s2[:100], cut_all=True, HMM=False,
                       use_paddle=True)  # 开不开HMM实际上没有影响
    can_eliminate = False
    for r in result:
        if conn in r:
            can_eliminate = True
            break
    if can_eliminate:
        return True # 结巴分出来的词里含有的，我们可以合并

    if r1 > 0.667 and r2 > 0.667:  # 含中文元素太少的不去
        return True
    return False


def eliminate_zh_space(flatten: list[str]) -> list[str]:
    """
    成句：
        对于中文，我们需要一个滑动窗口扫描每个字周围的字，
        由于双字词语最多，字越多的词语越少，我们需要一种函数来计算一个字和其他字的上下文相关度。
        我们仅删除字与字之间上下文相关度低的空格。或者这一步我们直接交给jieba

    """
    def merge(buf: list, segment: list):
        for i in segment:
            buf.append(i)
            while len(buf) >= 2 and can_concat_two_by_ruleset(buf[-2], buf[-1]):
                bck = buf.pop()
                buf[-1] += bck

    linebuf = []
    for line in flatten:
        seg = line.split() # 丢掉多余的空字符
        segbuf = []
        merge(segbuf, seg)
        linebuf.append(' '.join(segbuf))

    # linebuf2 = []
    # merge(linebuf2, linebuf)

    # return '\n'.join(linebuf2)
    return linebuf


def eliminate_zh_breakline_prework(flatten: str) -> dict:
    """统计字的上下文衔接度，可以分为用jieba分词后按词统计，也可以直接按字统计
    Args: 
        flatten (str): 没有分页符的整个文件的文本
    Returns:
        Dict[str, Counter[str, int]]: 上下文计数字典 {后面的词: {前面的词: 出现次数}}
    """
    CONTEXT_LENGTH = 1  # 超参
    SCORE = [1]  # 相关度赋分，保持长度与context_length一致

    near_word_counter = {}
    for line in flatten:
        # for cid, char in enumerate(line):
        #     if char in all_punctuation_set:
        #         continue
        #     char_stat = near_word.setdefault(char, {})
        #     for back_char_index in range(max(0, cid - context_length), cid):
        #         back_char = line[back_char_index]
        #         if back_char in all_punctuation_set:
        #             continue
        #         distance = cid - back_char_index
        #         char_stat[back_char] = char_stat.get(back_char, 0) + score[distance - 1]
        for zhseg in re.findall(IS_ALL_THIS_LANG['zh'], line):
            cut_list = jieba.lcut(zhseg, use_paddle=True)
            for wid, word in enumerate(cut_list):
                word_stat = near_word_counter.setdefault(word, Counter())
                for front_word_index in range(max(0, wid - CONTEXT_LENGTH), wid):  # 往前找
                    front_word = cut_list[front_word_index]
                    dist = wid - front_word_index
                    word_stat[front_word] += SCORE[dist - 1]
    return near_word_counter

def eliminate_zh_breakline_mainwork(flatten: list[str], near_word_counter: dict[str, Counter]) -> str:
    """清除断行
    """
    # CONCAT_THRESOLD = 1 # 超参，超过这个得分的我们合并


    linebuf = []
    prvline = ''
    for line in flatten:
        line = line.strip()
        if not line: # 丢掉多余的空换行
            continue
        # if '安全情况' in line:
            # print('breakpoint')
        # linebuf为空，或者两行中任意一行不含中文
        if not linebuf or not re.search(IS_ALL_THIS_LANG['zh'], line) or not re.search(IS_ALL_THIS_LANG['zh'], linebuf[-1]):
            linebuf.append(line)
            prvline = line
            continue
        s1 = linebuf[-1]
        s2 = line
        if re.match(LINEDOT_TOKEN, s2) or re.match(LINENO_TOKEN, s2): # 避免跟有序列表规则冲突
            linebuf.append(line)
            prvline = line
            continue
        # back_char = s1[-1]
        # front_char = s2[0]
        # 不处理标点符号
        # if back_char in PUNCTUATION_SET or front_char in PUNCTUATION_SET:
        score = 0 # 正表示删换行，负表示保留换行

        if zh_is_end_punctuation(s1):
            score -= 144
        if zh_isnot_end_punctuation(s1):
            score += 288

        if prvline:
            score += 3 * (min(60, len(prvline)) - 21) # 长度权重设高一些

        # 特判目录：阿拉伯数字和中文数字中的换行不处理
        # if back_char in string.digits and front_char in DIGITS['zh'] or \
        #         back_char in DIGITS['zh'] and front_char in string.digits:
        #     linebuf.append(line)
        #     continue
        if can_concat_two_by_ruleset(s1, s2): # 能成词的，有比较合并
            score += 999
        # 只看两个字接在一起
        # char_stat = near_word.setdefault(front_char, {}).get(back_char, 0)
        # if char_stat >= concat_thresold:
        #     linebuf[-1] += line
        # else:
        #     linebuf.append(line)

        back_word = jieba.lcut(s2, use_paddle=True)[-1]
        front_word = jieba.lcut(s1, use_paddle=True)[-1]
        word_stat = near_word_counter.setdefault(back_word, Counter()).get(front_word, 0)
        if word_stat == 0:
            score -= 80
        else:
            score += 100 * (word_stat - 1)

        # if word_stat >= CONCAT_THRESOLD:
        if score > 0:
            linebuf[-1] += line
        else:
            linebuf.append(line)
        prvline = line

    return linebuf


def read_int(s: str) -> int:
    """从s的开头开始读一段连续的数字"""
    x = 0
    for c in s:
        if c.isdigit():
            x = x * 10 + int(c)
        else:
            return x
    return x


def start(zh_text: str):
    """先去空格，再去换行"""
    flatten = cat_by_lineno(zh_text.split(PAGINATION_TOKEN))
    flatten = eliminate_zh_space(flatten)
    near_word_counter = eliminate_zh_breakline_prework(flatten)
    flatten = eliminate_zh_breakline_mainwork(flatten, near_word_counter)
    return '\n'.join(flatten)

    # return eliminate_zh_space("\n".join(mainwork(prework(zh_text))))
