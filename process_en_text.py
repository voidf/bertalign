import re
import itertools

import nltk
from helper import cat_by_lineno
from helper import match_lineno_seg
from helper import PAGINATION_TOKEN
from difflib import SequenceMatcher
# GLOBAL CONSTANTS
INDEX_TOKEN = '...'



def extract_sentences_from_single_file(filetext: list[str]) -> str:
    """
    此函数会尝试把属于单个文件里的意外被换行符断开的句子恢复回来，
    并且过滤掉部分分页带来的冗余信息。
    返回的字符串是整个文件已经去除了分页信息的文本串
    为了保证规则准确性，输入应该按文本的每行事先做好strip
    Args:
        filetext (list[str]): 按页分开的，来自于同一个文件的文本串
    Returns:
        str: 按如上描述清洗后的文本串
    Example:
        >>> extract_sentences_from_single_file(["Everything seemed to be\nalright.", "Cause you gave\nme whispers of\nlove all night."])
        "Everything seemed to be alright.\nCause you gave me whispers of love all night."
    """

    # 根据观察，有至少三个因素影响一行结尾的回车能不能被删掉
    # 1. 次行首字母是不是小写字母
    # 2. 本行末尾字符是不是句号
    # 3. 本行是不是约有50个字符

    flatten: list[str] = cat_by_lineno(filetext)
    outputs = [flatten[0]]
    for lineid, nextline in enumerate(flatten[1:]):
        prevline = outputs[-1]
        if (not nextline) or (not prevline): # 当两行中一行是空行，则拼接
            outputs[-1] += nextline
            # outputs.append(nextline)
            continue
        if match_lineno_seg(nextline): # 避免和cat_by_lineno规则冲突
            outputs.append(nextline)
            continue

        score = 0 # 正表示删换行，负表示保留换行
        if prevline[-1] in ('.', '?', '!', ';'):
            score -= 44
        if prevline[-1] == ',':
            score += 81

        score += min(60, len(flatten[lineid])) - 32

        # 加入nltk的条件，太长会严重影响性能，限制前一句最多100字符
        nextline2Bjoined = nextline[:100]
        joined = outputs[-1][-100:] + ' ' + nextline2Bjoined
        tokenized_by_nltk = nltk.sent_tokenize(joined)
        if len(tokenized_by_nltk) == 1:
            score += 200
        elif len(tokenized_by_nltk) >= 2:
            # 遍历结果，找到一个ratio和第二句差不多的
            maxratio = 0
            for token in reversed(tokenized_by_nltk):
                sm = SequenceMatcher(lambda x: x==' ', token, nextline2Bjoined, autojunk=True) # 0.6->0 0.9->200
                if sm.real_quick_ratio() < maxratio or sm.quick_ratio() < maxratio:
                    continue
                maxratio = max(maxratio, sm.ratio())
            score -= (maxratio - 0.6) * 666.7 # * 200 / 0.3
            # s1, s2 = tokenized_by_nltk
            # if s1 == prevline and s2 == nextline:
                # score -= 200
            # if is_likely(s1, outputs[-1]) and is_likely(s2, nextline):
                # score -= 200

        if nextline[0].islower():
            score += 83


        if score > 0:
            outputs[-1] = joined
        else:
            outputs.append(nextline)


    output = '\n'.join(outputs)

    return output






def start(text):
    return extract_sentences_from_single_file(text.split(PAGINATION_TOKEN))

   

