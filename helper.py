from collections import namedtuple
import itertools
import json
import os
import re
from typing import Dict, Optional, Union
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

def use_proxy():
    import socks
    import socket
    socks.set_default_proxy(socks.SOCKS5, '127.0.0.1', 7890)
    socket.socket = socks.socksocket

def read_secret(relative_path, hint=''):
    relative_path += '.secret'
    abs_path = my_path(relative_path)
    if not os.path.exists(abs_path):
        cmd = input(f'[{hint} {relative_path}] The secret file is required, your input will be saved in {abs_path}. \nNow please input:')
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(cmd)
        print(f'Your input is saved to {abs_path}, modify it if it is incorrect.')

    try:
        with open(abs_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        print(e)
        print(f'please put your token to {relative_path} in the root dir specify in WORKDIR_ABSOLUTE')
        print('current WORKDIR_ABSOLUTE:', WORKDIR_ABSOLUTE)
        raise


def chat(prompt: str):
    debug_prompt_engineering = '''I need your help to solve a breakline elimination problem,
given some text exported from PDF, 
some breakline may split the text as meaningful paragraghs but others could separate them unexpectly,
in this case, you should join adjacent lines if they can form a meaningful paragraph and replace the breakline symbols as spaces.
try to filter noises and keep as many meaningful info as you can, 
leave the indexing information and some lines that can not form a paragragh as it is, 
do not add more word to the input text, 
do not answer any other word except the task output,
format the resulting paragraphs as python list, and make sure it can use by python's eval with no error.
Here is the input text:

'''
    production_prompt_engineering = '''I need your help to solve a breakline elimination problem,
given some text exported from PDF, 
some breakline may split the text as meaningful paragraghs but others could separate them unexpectly,
in this case, you should join adjacent lines if they can form a meaningful paragraph and replace the breakline symbols as spaces.
leave the indexing information and some lines that can not form a paragragh as it is, 
do not answer any other word except the task output,
do not echo the processed text, 
just tell me the indexes of the breakline symbol you replaced with spaces, 
assume the first breakline symbol has the index 0,
and please separate the indices by comma.
Do not answer any characters except the comma separated index numbers.
Here is the input text:
'''

    import requests
    k = read_secret('openai_token')
    # inputs = debug_prompt_engineering + prompt
    inputs = production_prompt_engineering + prompt
    tokens = len(inputs.split())
    print('tokens len:', tokens)
    r = requests.post("https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + k
            },
            json={
                # "model": "text-davinci-003",
                "model": "gpt-3.5-turbo",
                # "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": inputs},
                    {"role": "assistant", "content": 'Output:\n'}
                    ],
                "temperature": 0, 
                "max_tokens": 4000 - int(tokens * 1.3)
            }
        )
    print(r.json())
    with open(my_path('chatgptoutputs.jsonl'), 'a', encoding='utf-8') as f:
        f.write(r.text)
    return r.json()['choices'][0]['message']['content']
    

if __name__ == "__main__":
    example = """United Nations A/AC.105/799
General Assembly
Distr.: General
4 December 2002  Original: English
V.02-60213 (E)    23202    24202
*0260213* Committee on the Peaceful    Uses of Outer Space
Report on the Third United  Nations/International Academy
of Astronautics Workshop on Sm all Satellites at the Service
of Developing Countries: Beyond Technology Transfer
(Houston, United States of America, 12 October 2002)
Contents
Paragraphs Page
I. Introduction ......................................................... 1-6 2
A. Background and objectives ........................................ 1-4 2
B. Attendance ..................................................... 5-6 2
II. Summary of presentations ............................................. 7-21 3
III. Conclusions and recommendations ...................................... 22-27 6
I. Introduction
A. Background and objectives
1. The Third United Nations Conference on  the Exploration and Peaceful Uses of
Outer Space (UNISPACE III) recommended, in ter alia, that the joint development,
construction and operation of a variety of small satellites offering opportunities to
develop indigenous space industry should be undertaken as a suitable project for enabling space research, technology demons trations and related applications in
communications and Earth observation.
1 Additional recommendations emanated
from the activities of the Technical Forum held at UNISPACE III.2 In accordance
with those recommendations, the Office for Outer Space Affairs of the Secretariat has substantially extended its existi ng cooperation with the Subcommittee on
Small Satellites for Developing Nations  of the International Academy of
Astronautics (IAA).
3
2. At its forty-fourth session, in 2001, the Committee on the Peaceful Uses of
Outer Space endorsed the programme of workshops, training courses, symposiums and conferences planned for 2002.
4 Subsequently, the General Assembly, in its
resolution 56/51 of 10 December 2001, endor sed the United Nations Programme on
Space Applications for 2002.
3. At the 1999 meeting of the IAA Subcommittee, it was agreed that the fifty-
first International Astronauti cal Congress, which was to be held in Rio de Janeiro,
Brazil, from 2 to 6 October 2000, would be an ideal opportunity to review the status and advancement of programmes in Latin Am erica. It was further agreed that the
Workshop should be open to participants from other regions, but that the situation in Latin America would be used as an example of how developing countries could benefit from small satellites and that it should form the core of the discussion. The report of the first United Nations/IAA Workshop (A/AC.105/745) was submitted to the Scientific and Technical Subcommittee at its thirty-eighth session, in 2001. Based on the positive response from particip ants and from States members of the
Committee, it was decided that the second Workshop, to be held in 2001, should encourage the development of small satellite technology in Africa. The Workshop was held in Toulouse, France, on 2 October 2001 and the corresponding report (A/AC.105/772) was submitted to the Scientific and Technical Subcommittee at its thirty-ninth session in 2002.
4. The United Nations/IAA Workshop on Small Satellites at the Service of
Developing Countries: Beyond Technology Transfer was held in Houston, United States of America, on 12 October 2002. It was the third Workshop organized jointly by the Office for Outer Space Affairs and the IAA Subcommittee on Small Satellites
for Developing Nations within the framework of the International Astronautical Congress.
B. Attendance
5. The Workshop was an integral part  of the World Space Congress and was
attended by as many as 85 registered Congress participants. Many of those attending the Workshop had also attended the Unite d Nations/International Astronautical
Federation Workshop on Space Solutions for Global Problems: Building Working
Partnerships with All Stakeholders in Human Security and Development (A/AC.105/798). The sponsors of the workshop (the United Nations Educational, Scientific and Cultural Orga nization, the European Space Agency and the National
Aeronautics and Space Administration (NASA) of the United States provided financial support to selected participants from developing countries.
6. One of the objectives of the Workshop was to review the utilization of small
satellites not only for the purpose of tec hnology transfer, but also as a useful
contribution to the development of countri es and to scientific or application
programmes; the Workshop was conducted in the light of the recommendations of the previous workshops. The Workshop was al so attended by several participants of
previous workshops who provided valuable continuity and were able to assess the progress that had been made during the series of workshops.
II. Summary of presentations
7. In a brief introduction, the Workshop co-chairmen gave an overview of the
results achieved at workshops held at UNISPACE III, in Rio de Janeiro and in Toulouse. Seven papers were  then presented and discussed, most of which dealt
with applications in the field of remote sensing and Earth observation.
8. The first paper dealt with AlSAT-1,  which was the first Algerian national
satellite. Developed in partnership with the United Kingdom of Great Britain and Northern Ireland as part of a know-how and technology transfer programme, the satellite was to be the first to be launched by several countries as part of a disaster monitoring constellation (DMC). The cooperative programme involved Algeria, China, Nigeria, Thailand, Turkey, the United Kingdom and Viet Nam. Satellites from the seven countries were to be put into the same orbit in order to form the first international constellation de dicated to monitoring natural and man-made disasters.
They would enable the seven countries to have daily access to global images for disaster mitigation, national remote se nsing applications, and space commercial
exploitation and would facilitate interna tional cooperation be tween developed and
developing countries.
9. As part of DMC, AlSAT-1 would co ntribute to mitigating natural and man-
made disasters through early warning, event monitoring and analysis. When the satellite was not being used for DMC purposes, it would be monitored and
controlled for national applications. Algeria was a big country, the second largest on the African continent, and had an ar ea greater than 2.5 million square
kilometres (km). It needed to monitor agricultural land use, industrial and marine pollution and support cartography for infrastructure such as road and rail networks which could best be done by the use of satellites. Another specific regional
application was the monitoring of the accel erating desertification that was occurring
on the boundaries of the Sahara.
10. AlSAT-1was the first satellite to be launched as part of the space programme
that Algeria intended to carry out over the following decade. The programme was designed to support Algeria’s development needs and for purposes of education, marine and atmospheric pollution, telecommunications, utilization of natural resources, weather and climate applications, urban and rural infrastructure, and land
use management and to assist in resolving other local-level resource problems. As
part of a sustainable space programme, the Algerian Centre national des techniques
spatiales was already planning the launc h of a second spacecraft, AlSAT-2.
11. Nigeria was also developing its first mi crosatellite, NigeriaSAT-1, as part of
DMC. The satellite was part of a Nati onal Space Research and Development
Programme that was being implemente d by the National Space Research and
Development Agency (NASRDA). The programme constituted an important
component of the national strategy for socio-economic development through space applications. Among the Agency’s objectives were the development of indigenous capabilities in the major areas of space sc ience and technology and the use of those
capabilities as tools for natural resource ma nagement, infrastructure development,
environmental monitoring and sustainable development. The paper presented the policy, objectives and institutional fram ework, as well as the mandate of the
Agency. The NASRDA programme was built around the following themes: development of human resources and capac ity-building; management of natural
resources; study of the Earth and its environment; defence, national security and law enforcement; space communication applica tions; and education and training. The
promotion of international cooperation was iden tified as an integral part of the space
programme in Africa, in par ticular within the Economic Community of West African
States (ECOWAS).
12. The NigeriaSAT-1 project was being developed in cooperation with the United
Kingdom and included technology transfer and training components. Further plans to develop a communication satellite, Ni geriaSAT-2, were in progress; it was
recognized that ineffective communications re presented one of the greatest barriers
to socio-economic development and NigeriaSAT-2 would be designed to contribute to providing an adequate telecommunications system throughout Nigeria and regional coverage to ECOWAS countries.
13. The third paper, from South Africa, dealt with the digital divide in Africa. A
core objective of the New Partnership for African Development, a mandated programme of the African Union, was to gi ve impetus to Africa’s development by
bridging existing gaps in priority sectors, one of which was information and communication technologies and the pressing need to bridge the digital divide. The paper expressed the view that small and micro satellites provided one of the most appropriate instruments for meeting that objective; in fact, several countries had
already launched or were developing sma ll satellites (Algeria and Nigeria, as
presented at the Workshop; South Afri ca with the SUNSAT satellite), which
provided a basis for further developments.
14. The successful launch and opera tion of the SUNSAT micro satellite
demonstrated that the technology base for Earth-observation applications, environmental, agricultural and agro-meteorological, could be established using a very small, high-value satellite platform. It was proposed to develop an African
resource management (ARM) constellation through an African cooperative programme. The application of those satel lites could contribute to  meeting the needs
of African countries in a sustainable manner and to addressing problems such as the “brain drain”, lack of access to space technology and data, poverty and food insecurity, disasters, poor infrastructure, refugees and unsustainable development. With the current satellite developments , space engineering capabilities were
becoming accessible within Africa itself and a commitment to long-term research
and development could only be sustained by repeatable development and utilization
of technology and know-how. The establishment of an ARM constellation would contribute to the fulfilment of one of the key aims of the New Partnership for African Development.
15. The fourth paper, from Indonesia, pres ented the design of a new micro satellite
for resource monitoring, Ganesyasat-CXM. The satellite would have equatorial low-Earth orbit for optimum temporal resolution for the main environmental monitoring mission.
16. Indonesia was a maritime country, comprising over 14,000 islands spread
along one eighth of the equator, with some 81,000 km of coastline, approximately 1.9 million square km of land, 3.1 milli on square km of territorial sea and
2.7 million km of exclusive economic zone. It s maritime status was a driving factor
for development activities and business ventures and those, together with its need to manage a wealth of natura l resources, both terrestria l and marine, as well as
agriculture and forestry, justified use of space technology.
17. The plan to launch a satellite was therefore conceived on the premise that
space technology could make significant c ontributions to solving problems related
to national economic development. It would be used for the education of students in spacecraft design and manuf acturing. When in orbit, it would contribute to
environmental observations and geogra phical information and would support
scientific studies associated with meteorological and volcanic activity surveillance.
18. The fifth paper concerned the Arge ntine SAC-C (Satélite de Aplicaciones
Científicas) Satellite, which was an in ternational Earth-observing satellite
developed by the Argentina Comisión Nacional de Actividades Espaciales in partnership with the United States, with additional support in instrumentation and
satellite development from Brazil, Denmar k, France and Italy. The satellite was
entirely built and assembled in Argentina. There were 10 instruments on board SAC-C that carried out studies on the ev aluation and evolution of desertification
processes, the identification and prediction of agricultural production, the monitoring of flood areas, as well as studies in coastal and freshwater areas. Additional scientific objectives were to monitor the condition and dynamics of the terrestrial and marine biosphere and environment, to contribute to a better understanding of the Earth’s magnetic field and related Sun-Earth interactions and
to develop and utilize new Global Positioni ng System (GPS) techniques to measure
atmospheric phenomena on a global scal e for the study of weather, seasonal,
interannual and long-term climate change.
19. The SAC-C satellite was launched in November 2000 and was part of the
“Morning Constellation” along with three United States satellites: Landsat-7, EO-1 and Terra. The creation of such a cons tellation permitted the quasi-simultaneous
acquisition from the four satellites of im ages of various geometric and spectral
resolutions in different spectral bands, the carrying out of autonomous navigation experiments and the testing of the G PS satellite constell ation capabilities for
atmospheric studies, navigation and attitude and orbit control. The main application
areas were hydrology, desertification, urba n planning, precision farming, forestry,
ecology, atmospheric and ionospheric studies and cloud properties. Using data from the four satellites, interesting results were obtained on land use, native forest
resources and floods and fires, the latter being the most important potential
hazardous events in Argentina.
20. The sixth paper, from Brazil, presen ted the novel application of the Brazilian
data collection satellites SDC 1 and 2 to pr ecision farming of orange crops. For that
application, the data collecting platform s located on the ground would collect data
related to soil moisture and the height of the fruit, which were important parameters
for the flowering process and consequently the production of fruit itself; those data would be transmitted to the user via the SCD satellites. Such an application was valid only for perennial crops, but the app lication could also be extended to other
types of agricultural data for government or private use.
21. The final paper presented a small scie ntific satellite project on space weather
monitoring, to be developed jointly by Brazil and the Russian Federation. The Russian Federation had had experience in th at area with the Interball series of
satellites. A joint Russian-Ukrainian mission combining a Russian Interball satellite
and a Ukrainian Prognoz satellite was planne d. Brazil could provide a third satellite
on a highly elliptical orbit. By using the cons tellation of satellites in different orbits,
it would be possible to monitor interplanetary and magnetospheric phenomena with variable spatial and temporal characteristics. It was expected that such data could be used to improve space weather forecasting and monitoring.
III. Conclusions and recommendations
22.  The Workshop clearly demonstrated, once again, that there were tremendous
spin-offs to be gained from introducin g space activities through a small satellite
programme.
23. The participants of the Workshop recognized that small satellites were a useful
tool for acquiring and developing tech nology and contributing to education and
training. The Workshop stressed the importance of placing the main focus on applications that provide sustainable eco nomic benefits for developing countries.
24. In the presentations it was emphasized that practical results had already
demonstrated that small satellites were effective in addressing regional problems.
New programmes had been presented and were  expected to provide benefits such as
those arising from remote sensing, especially  in such fields as disaster mitigation,
agriculture, desertification, forest mon itoring and infrastructure development.
25. The participants also recognized that small satellit e projects were promoting
through bilateral or multilateral agreemen ts international cooperation within a
region or worldwide. The Workshop recognized that small satellite projects could result in fruitful cooperation between different countries in the planning, implementation and maintenance of a conste llation of satellites, as well as in the
effective utilization of the data acquired. The participants recognized that such an approach could be a useful means of sharing satellite development cost and
information data.
26. The Workshop recognized that, within a country, a small satellite programme
could stimulate interest in science and technology, enhance quality of life and the
quality of education, promote research an d development and resu lt in better linkages
between government agencies, educational institutions and industries. The
participants therefore emphasized the need for greater awareness among the public
and among decision makers of the benefits of space programmes.
27. The participants of the Workshop recognized that the proposals made during
UNISPACE III were fully applicable, but th ey made or reconfirmed the following
additional conclusions and recommendations:
(a) The Workshop recognized that avenues of international cooperation
should continue to be explored in order to foster the use of small satellite systems for the benefit of developing countries, including through the promotion of regional projects. For that purpose, the Workshop recommended that coordinated action be continued to identify significant problems th at were common to different countries
in a region and that could be addressed with the help of small satellite technology. The Workshop also recommended that partnerships be developed between regions with common needs, such as the equatorial regions of different continents;
(b) Efforts had been made to develop space systems devoted to improving
the quality of life in developing countries. To provide maximum economic and social benefits to the populations of such countries, the Workshop recommended that programmes be established in such a manner as to ensure continuity and sustainability;
(c) The Workshop highlighted, in particular, the growing importance of
Earth observation programmes for developi ng countries and the benefits of
international cooperative efforts. The Workshop therefore recommended that long-term strategic programmes be developed to  ensure the sustainable acquisition and
processing of the data needed for monitori ng the environment and natural resources,
for the mitigation of man-made or natural di sasters, as well as for decision-making;
(d) The Workshop recognized the benefits of small satellite programmes in
the acquisition, development and application of space science and technology, and the associated development of a knowledge base and industrial capacity. The Workshop therefore recommended that space activities be an integral part of any national programme devoted to the acqu isition and development of technology and
capacity-building;
(e) The Workshop confirmed that it recognized the importance of space
development in education curricula, especially for motivating and training students. In line with the recommendations of UNISPACE III, the Workshop recommended that each country recognize the important role that space assets could play in education and the need to incorporate space science and technology in curricula;
(f) Finally, the Workshop emphasized the need to develop among the general
public as well as among decision makers an awareness of the potential benefits of space technology applications. In particular, it recognized the important role that a
dedicated organization or agency could play in the definition and implementation of a space programme. The Workshop recommended that every country or group of countries consider the attainment of a minimum level of space capabilities as they could be invaluable in enhancing socio-economic development and the quality of life of populations.
Notes
1  Report of the Third United Nations Conference on the Exploration and Peaceful Uses of Outer
Space, Vienna, 19-30 July 1999  (United Nations publication, Sales No. E.00.I.3), chap. I,
resolution 1, annex, para. 32 (b).
2  Ibid., annex III.
3  The purpose of the IAA Subcommittee on Small Satellites for Developing Nations is to assess
the benefits of small satellites for developing countries and to develop awareness on the subject
in both developed and developing countries. The IAA Subc ommittee publishes its findings and
disseminates relevant informati on through workshops and symposiu ms. In order to realize its
goals, the IAA Subcommittee c ooperates with: the United Nati ons and its Committee on the
Peaceful Uses of Outer Space; the Internati onal Astronautical Federa tion and its Committee for
Liaison with International Orga nizations and Developing Nations ; and the International Space
University.
4  Official Records of the General Assembly, Fifty-seventh Session, Supplement No. 20  (A/57/20),
para. 54.""".splitlines()

    use_proxy()
    for i in range(3):
        chat('\n'.join(example[i*20:i*20+40]))
    # chat('\n'.join(example[0:20]))
    # chat('\n'.join(example[10:30]))
    # chat('\n'.join(example[30:40]))
    # chat('\n'.join(example[40:50]))

