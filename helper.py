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


def filter_leading_and_tail_blank_lines(lines: list[str]) -> list[str]:
    """去除前导空行和尾随空行（其实应该直接用strip）"""
    newlines = []
    for line in lines:
        line = line.strip()
        if not line and not newlines: # 去前导空行
            continue
        newlines.append(line)
    while newlines and not newlines[-1]: # 去尾随空行
        newlines.pop()
    return newlines

def make_banner(record: str) -> str:
    divider = '=' * 10 + '\n'
    return  divider + record + '\n' + divider


def make_filter_log(filtered: str, record: str | int, lang: str, page: str | int, reason: str):
    """将过滤的内容写到log里方便分析"""
    with open(r'C:\Users\Administrator\Documents\parallel_corpus_mnbvc\alignment\bertalign\filter_log.jsonl', 'a', encoding='utf-8') as f:
        json.dump({'record': str(record), 'lang': lang, 'page': str(page), 'reason': reason, 'filtered': filtered}, f)
        f.write('\n')

def dump_row(row):
    """调试用，输出中间结果到文件，row是map的DatasetDict"""
    for lang in LANGS:
        with open(r'C:\Users\Administrator\Documents\parallel_corpus_mnbvc\alignment\bertalign\pre' + f'/dbg_{lang}.txt', 'a', encoding='utf-8') as f:
            f.write(make_banner(row['record']) + row[lang])

def dump_align_result_to_file(record: str, result: dict):
    Path(r'C:\Users\Administrator\Documents\parallel_corpus_mnbvc\alignment\bertalign\done').mkdir(parents=True, exist_ok=True)
    for lang in result:
        with open(r'C:\Users\Administrator\Documents\parallel_corpus_mnbvc\alignment\bertalign\done' + f"/{lang}.txt", "a", encoding="utf-8") as f:
            f.write(make_banner(record) + result[lang])


def read_int(s: str) -> int:
    """从s的开头开始读一段连续的数字"""
    x = 0
    for c in s:
        if c.isdigit():
            x = x * 10 + int(c)
        else:
            return x
    return x


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


LINENO_TOKEN = re.compile(r'^\d+\. ') # 标号后面还是要跟一个空格
LINEDOT_TOKEN = re.compile(r'^• ')

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
        m = re.match(LINENO_TOKEN, line)
        if m:
            g = m.group(0)
            match_infos.append((read_int(g), lineid))
        m = re.match(LINEDOT_TOKEN, line)
        if m:
            g = m.group(0)
            match_infos.append((-114514, lineid))

    for idx, (linecounter, lineid) in enumerate(match_infos[1:]):
        # 相邻两个识别头标号连续，或者都是点标号，则中间行的\n可以删掉（换成空格，将两段话拼在一起）
        prevcounter, previd = match_infos[idx]
        if linecounter == prevcounter + 1 or linecounter == prevcounter == -114514:
            line_marker.extend(range(previd, lineid - 1))

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