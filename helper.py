from collections import namedtuple
import itertools
import json
import re
import Levenshtein
import string
from pathlib import Path

LANGS = ['zh', 'fr', 'es', 'ru', 'en', 'ar'] # 全语种
LANGS_WITHOUT_AR = ['zh', 'fr', 'es', 'ru', 'en']
PAGINATION_TOKEN = '\n----\n'

LANGS = LANGS_WITHOUT_AR

WORKDIR_ABSOLUTE = r'C:\Users\Administrator\Documents\parallel_corpus_mnbvc\alignment\bertalign'

easy_version = '5'
PREPROCESS_DIR = f'pre{easy_version}'
ALIGNED_DIR = f'done{easy_version}'
FILTER_LOG = f'filter_log{easy_version}.jsonl'
ERROR_LOG = f'errors_log{easy_version}.jsonl'
ALIGN_LOG = f'align_log{easy_version}.jsonl'

def cat(*args): 
    return '/'.join(args)

def my_path(*args):
    return cat(WORKDIR_ABSOLUTE, *args)

def ensure_dirs():
    for d in [PREPROCESS_DIR, ALIGNED_DIR]:
        Path(my_path(d)).mkdir(parents=True, exist_ok=True)
    

def make_banner(record: str) -> str:
    divider = '=' * 10 + '\n'
    return  divider + record + '\n' + divider


def make_filter_log(filtered: str, record: str | int, lang: str, page: str | int, reason: str):
    """将过滤的内容写到log里方便分析"""
    with open(my_path(FILTER_LOG), 'a', encoding='utf-8') as f:
        json.dump({'record': str(record), 'lang': lang, 'page': str(page), 'reason': reason, 'filtered': filtered}, f)
        f.write('\n')

def align_logger(info: str):
    print(info)
    with open(my_path(ALIGN_LOG), 'a', encoding='utf-8') as f:
        f.write(info + '\n')

def dump_row(row):
    """调试用，输出中间结果到文件，row是map的DatasetDict"""
    for lang in LANGS:
        with open(my_path(PREPROCESS_DIR, f'dbg_{lang}.txt'), 'a', encoding='utf-8') as f:
            f.write(make_banner(row['record']) + row[lang])

def dump_align_result_to_file(record: str, result: dict):
    Path(my_path(ALIGNED_DIR)).mkdir(parents=True, exist_ok=True)
    for lang in result:
        with open(my_path(ALIGNED_DIR, f"aligned_{lang}.txt"), "a", encoding="utf-8") as f:
            f.write(make_banner(record) + result[lang])




def is_likely(s1: str, s2: str, thresold=3) -> bool:
    """
    这个函数以两个字符串的编辑距离为标准决定两个字符串是否相似。
    （仅用于判断这段文本是否可以被当做目录索引文本而删除。）
    如果它们之间的编辑距离大于EDIT_DISTANCE_THRESOLD，则判为不相似。

    为了优化运行效率，在计算编辑距离之前，先做了两个剪枝：
    如果两个字符串长度差超过EDIT_DISTANCE_THRESOLD，则判为不相似。
    如果两个字符串顺序无关的字符编辑距离超过EDIT_DISTANCE_THRESOLD，则判为不相似。

    Args:
        s1 (str)
        s2 (str)
        thresold (int) 相似阈值，编辑距离大于这个值会返回False，默认为3

    Returns:
        bool: s1和s2是否相似

    Example:
        >>> is_likely("kit", "sitting")
        False
        >>> is_likely("flaw", "lawn")
        True
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    if len(s2) - len(s1) > thresold: # 优化相当大的O(1)剪枝
        return False
    
    # O(n)统计字符，进一步剪掉一些不必要用n^2编辑距离的情况，实测625s优化到22s
    char_distance = 0
    d = {}
    for s in s1:
        d[s] = d.get(s, 0) + 1
    for s in s2:
        d[s] = d.get(s, 0) - 1
    positive = 0
    negative = 0
    for v in d.values():
        if v > 0:
            positive += v
        else:
            negative += - v
    char_distance = max(positive, negative)
    if char_distance > thresold:
        return False
    # 编辑距离
    edit_distance = Levenshtein.distance(s1, s2)
    if edit_distance > thresold:
        return False

    return True

def read_int(s: str) -> int:
    """从s的开头开始读一段连续的数字"""
    x = 0
    for c in s:
        if c.isdigit():
            x = x * 10 + int(c)
        else:
            return x
    return x

def read_back_int(s: str) -> int:
    """读最后一个.后的数字"""
    x = 0
    for c in s:
        if c.isdigit():
            x = x * 10 + int(c)
        elif c == '.':
            x = 0
    return x


ROMAN_VAL = {
    'I': 1,
    'V': 5,
    'X': 10,
    # 'L': 50,
}

def read_roman(s: str) -> int:
    """读罗马数字"""
    prev = 0
    curr = 0
    num = 0
    for i in reversed(s):
        if i in ROMAN_VAL:
            curr = ROMAN_VAL[i]
            if curr < prev:
                num -= curr
            else:
                num += curr
            prev = curr
    return num

def read_en_letter(s: str, begin_char='a') -> int:
    for i in s:
        o = ord(i) - ord(begin_char)
        if 0 <= o <= 25:
            return o
    return -2

CHINESE_NUM_DICT = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
                    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}

CHINESE_UNIT_DICT = {'十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}

def read_chinese(s: str) -> int:
    """读汉字"""
    num = 0
    unit = 1
    for digit in reversed(s):
        if digit in CHINESE_UNIT_DICT:
            if CHINESE_UNIT_DICT[digit] < unit:
                unit = CHINESE_UNIT_DICT[digit]
                num += unit
            else:
                unit = CHINESE_UNIT_DICT[digit]
        elif digit in CHINESE_NUM_DICT:
            num += CHINESE_NUM_DICT[digit] * unit
    return num


# LINENO_TOKEN = re.compile(r'^\d+\. ') # 标号后面还是要跟一个空格
# LINEDOT_TOKEN = re.compile(r'^• ')
LINENO_SEG_TOKENS = [
    (re.compile(r'^\d{1,3}\. '), read_int), # 有序列表，阿拉伯数字，很少有上千的，不写+而是{1,3}，避免错误匹配一些年份 1.
    (re.compile(r'^• '), lambda x: None), # 无序列表 •
    (re.compile(r'^\d{1,2}\.\d{1,2} '), read_back_int), # 第二类有序列表，阿拉伯数字带小标号 1.1
    (re.compile(r'^[IVX]{1,5}\. '), read_roman), # 有序列表，罗马数字 I.
    (re.compile(r'^\([a-z]\) '), read_en_letter), # 有序列表，括号小写英文 (a)
    (re.compile(r'^[a-z]\) '), read_en_letter), # 有序列表，半括号小写英文 a)
    (re.compile(r'^\d{1,3}\) '), read_int), # 有序列表，半括号数字 1)
    (re.compile(r'^\(\d{1,3}\) '), read_int), # 有序列表，全括号数字 (1)
    (re.compile(r'^[A-Z]\. '), lambda x: read_en_letter(x, 'A')), # 有序列表，大写英文标号 A. 
    (re.compile(r'^[一二三四五六七八九十]{1,3}、'), read_chinese), # 汉字有序列表 一、 
    (re.compile(r'^[一二三四五六七八九十]{1,3}\. '), read_chinese), # 汉字有序列表 一. 
    (re.compile(r'^\([一二三四五六七八九十]{1,3}\) '), read_chinese), # 第二类汉字有序列表 (一)
]

MatchedLinenoInfo = namedtuple('MatchedLinenoInfo', ['rule_id', 'int_index'])
def match_lineno_seg(line: str):
    """尝试跟列表规则组进行匹配，匹配不成功返回None，成功则返回一个MatchedLinenoInfo，line必须在传入前做strip
    int_index为None时，表示无序列表
    """
    for rule_id, (rule_pattern, process_func) in enumerate(LINENO_SEG_TOKENS):
        m = re.match(rule_pattern, line)
        if m:
            return MatchedLinenoInfo(rule_id, process_func(m.group(0)))
    return None

def cat_by_lineno(pages: list[str])-> list[str]:
    """根据有序列表标号去回车，过此函数后文本会合页，按页去噪应该早于此函数完成
    Args:
        filetext (list[str]): 按页分开的，来自于同一个文件的文本
    Returns:
        list[str]: 按回车分开的行文本
    """
    outputs = []
    match_infos = [] # 存(int数字列表号, int文件行号) 这样的二元组
    line_marker = [] # 可以去掉换行的行数
    
    flatten = list(line.strip() for line in itertools.chain(*[page.split('\n') for page in pages]))
    for lineid, line in enumerate(flatten):
        m = match_lineno_seg(line)
        if m:
            match_infos.append((m.rule_id, m.int_index, lineid))


    for idx, (rule_id, linecounter, lineid) in enumerate(match_infos[1:]):
        # 相邻两个识别头标号连续，或者都是点标号，则中间行的\n可以删掉（换成空格，将两段话拼在一起）
        prev_rule_id, prevcounter, prev_lineid = match_infos[idx]
        if prev_rule_id == rule_id:
            if linecounter is None or linecounter == prevcounter + 1:
                line_marker.extend(range(prev_lineid, lineid - 1))

    line_marker.reverse() # 反转，使标号满足递减序。

    for lineid, line in enumerate(flatten):
        while line_marker and line_marker[-1] < lineid - 1:
            line_marker.pop()

        if line_marker and lineid - 1 == line_marker[-1]:
            line_marker.pop()
            outputs[-1] += ' ' + line
        else:
            outputs.append(line)
    return outputs

WHITESPACES = set(string.whitespace.replace('\n', ''))
def filter_duplicated_whitespaces(src: str) -> str:
    """去噪：
        1. 如果换行符跟其它空格字符相连，这些字符替换成换行符
        2. 连续出现空格字符的，替换成其中一个空格字符"""
    buf = []
    newline = 0
    space = None
    for i in src:
        if i == '\n':
            newline += 1
        elif i in WHITESPACES:
            space = i
        else:
            if newline:
                buf.append('\n' * newline)
            elif space:
                buf.append(space)
            newline = 0
            space = None
            buf.append(i)
    if newline:
        buf.append('\n' * newline)
    elif space:
        buf.append(space)
    return ''.join(buf)