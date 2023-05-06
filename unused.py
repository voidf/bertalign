

VECTORS = []
SOURCE_TEXT = []
SOURCE_LINEID = []
MARKED = []



model = SentenceTransformer('LaBSE')

LANG = 'en'

# VEC_DIR = './linevec/' + LANG
VEC_DIR = r'C:\Users\Administrator\Documents\parallel_corpus_mnbvc\alignment\bertalign/linevec/' + LANG

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
    
def main(row):
    Path(VEC_DIR).mkdir(parents=True, exist_ok=True)
    rec = row['record']
    raw = row[LANG]
    lines = raw.split('\n')
    rec_filename = VEC_DIR + f'/{rec}.npz'
    if not os.path.exists(rec_filename):
        lines = raw.split('\n')
        vec = model.encode(lines, device='cuda:0')
        np.savez_compressed(rec_filename, vec)
        # print(vec)
    else:
        vec = np.load(rec_filename)['arr_0']
    assert len(lines) == len(vec)
    for lineid, (rawtext, vectorized) in enumerate(zip(lines, vec)):
        # EMBEDDING_CACHE[rawtext] = vectorized
        SOURCE_TEXT.append(rawtext)
        VECTORS.append(vectorized)
        SOURCE_LINEID.append(lineid)

# 还没形成用ai来做分页去噪的框架，先留档备份
def make_marked_file():
    npfn = VEC_DIR + '/VECTORS.txt'
    textfn = VEC_DIR + '/SOURCE_TEXT.txt'
    linefn = VEC_DIR + '/SOURCE_LINEID.txt'
    markedfn = VEC_DIR + '/MARKED.txt'

    projfn = VEC_DIR + '/PROJECTED.txt'

    from sklearn.decomposition import PCA
    pca = PCA(n_components=20)
    projected = pca.fit_transform(VECTORS)
    with open(projfn, 'w', encoding='ascii') as f:
        for x in projected:
            np.savetxt(f, x, fmt='%.2e', delimiter=',', newline='\t')
            f.write('\n')



    with open(npfn, 'w', encoding='ascii') as f:
        for x in VECTORS:
            np.savetxt(f, x[:10], fmt='%.2e', delimiter=',', newline='\t')
            f.write('\n')
    with open(textfn, 'w', encoding='utf-8') as f:
        f.write('\n'.join(SOURCE_TEXT))
    with open(linefn, 'w', encoding='utf-8') as f:
        f.write('\n'.join(map(str, SOURCE_LINEID)))
    with open(markedfn, 'w', encoding='ascii') as f:
        for x in range(len(VECTORS)):
            f.write('\n')
                    

# 算力开销过大
# def visualize():
#     from sklearn.cluster import DBSCAN
#     cached_filename = VEC_DIR + '/dbscan0.1.pkl'
#     if not os.path.exists(cached_filename):
#         dbscan = DBSCAN(eps=0.1, min_samples=1000).fit(np.array(VECTORS[:100000]))
#         with open(cached_filename, 'wb') as f:
#             pickle.dump(dbscan, f)
#     else:
#         with open(cached_filename, 'rb') as f:
#             dbscan = pickle.load(f)
#         for vec, text, lineid in zip(VECTORS, SOURCE_TEXT, SOURCE_LINEID):
#             d = dbscan.predict(vec)
#             print(text,)


        # 第一次过滤：分页符
        # if pageid == 0 and pageline:
        #     found = None
        #     for lineid, line in enumerate(pageline):
        #         if re.search(BEGIN_TOKEN, line) or re.search(BEGIN_TOKEN2, line): # 第一页中，在BEGIN_TOKEN之后的才是正文内容
        #             found = lineid
        #             break
        #     if found is not None:
        #         pageline = pageline[found:]



# 分页过滤的正则，由于例外太多，现在不使用，后续可以考虑用来兜底
pagination_info_re = [
    r'([a-zA-Z0-9\.]{1,13}/){2,5}[A-Za-z0-9\.]{1,13}', # 文件码
    r'^([0-9/]{1,8} ){0,1}[0-9-]{1,8}( [/0-9]{1,8}){0,1}$', # 文件码
    r'^(\([PartVol\.]{1,4} [IVX]{1,4}\)\*?)$', # 拿掉Part、Vol
    r'^Article [IVX]{1,4}$', # 拿掉Part、Vol
    r'^\*[0-9]+\*$',
    r'^[0-9]+-[0-9]+ \(E\)$',
]

# 目录应该是很好的分段依据，不应该去掉
def filter_index_title(file_index_titles: list[str], page: str) -> str:
    """把正文里的目录索引条目拿掉
    
    Args:
        file_index_titles (list[str]): 预处理得到的目录页的索引条目
        page: 源文件一页的文本内容
        
    Returns:
        str: 去掉了疑似索引条目的行的此页文本
    """

    filtered_page = []
    unmatched = deque() # 索引条目可能一行写不下，用一个队列来处理
    for line in page.split('\n'):
        line = line.strip()
        matched = False
        for cid, file_index_title in enumerate(file_index_titles): # 每个都for保证出现过的都拿掉，is_likely加了剪枝还不算慢
            if is_likely(file_index_title, line):
                while unmatched:
                    filtered_page.append(unmatched.popleft())
                matched = True
                # print(file_index_title, 'cid:', cid)
                filtered_page.append('\n====' + file_index_title +'====\n')
                break

            if unmatched:
                back_line = unmatched.pop() # 这个逻辑只处理最多两行文本
                if is_likely(back_line + ' ' + line, file_index_title):
                    while unmatched:
                        filtered_page.append(unmatched.popleft())
                    matched = True
                    # print(file_index_title, 'cid:', cid)
                    filtered_page.append('\n====' + file_index_title +'====\n')
                    break
                unmatched.append(back_line)

        if not matched:
            unmatched.append(line)
            while len(unmatched) > 1: # 三行和以上的索引条目非常少见，所以这里写1，如果有需要可以改大，但上面的组合逻辑也要改
                filtered_page.append(unmatched.popleft())
    while unmatched:
        filtered_page.append(unmatched.popleft())
    return '\n'.join(filtered_page)

# 这个规则能处理的文件太有限，不使用
    # 将The meeting rose at ...后一直到The meeting was called to order...中间的部分去掉
    # inputs: list[str] = outputs
    # outputs = []
    # accept_line = True
    # for line in inputs:
    #     if accept_line:
    #         if re.search(ROSE_TOKEN, line) or re.search(ROSE_TOKEN2, line):
    #             accept_line = False
    #         outputs.append(line)
    #     else:
    #         if re.search(BEGIN_TOKEN, line) or re.search(BEGIN_TOKEN2, line):
    #             accept_line = True
    #             outputs.append(line)

# 尽量不要过滤
def procedure(row):
    """main procedure for mapping"""
    filtered_pages = []
 
    lang_match_file_content = row.split('\n----\n')
    file_index_titles = []
    for pageid, page in enumerate(lang_match_file_content):
        lines = []
        dot_count = 0
        pageline = page.split('\n')

        for lineid, line in enumerate(pageline):
            line = line.strip()
            lines.append(line)
            if INDEX_TOKEN in line:
                dot_count += 1

            if dot_count >= 4: # 有大于4行三个点的我们认为是目录页，用特别的处理方式或者先跳过
                unmatched = []
                other_buffer = []

                for line in lines:
                    line = line.strip().replace('\ufffe', '-') # 瞪眼法得，\ufffe应该是连词符-
                    if INDEX_TOKEN in line:
                        title: str = line.split(INDEX_TOKEN, 1)[0].strip()
                        done = 0
                        # 预处理目录页，统计目录索引条目
                        # 有个特征是目录索引总是有一个带.的标号
                        for rid in range(len(unmatched), max(len(unmatched) - 4, -1), -1): # 最多处理连续4行的目录索引
                            concat_title = ' '.join([*unmatched[rid:], title])
                            dot_pos = concat_title.find('.')
                            if dot_pos != -1 and dot_pos < 6: # .出现的地方如果太靠后，我们不要
                                file_index_titles.append(concat_title)
                                done = 1
                                break # 没找到就取title
                        if not done:
                            file_index_titles.append(title)
                        other_buffer.extend(unmatched)
                        unmatched.clear()
                    else:
                        unmatched.append(line)
                other_buffer.extend(unmatched)
                lang_match_file_content[pageid] = '\n'.join(other_buffer) # 拿掉目录页
            else:
                dst = '\n'.join(lines)
                lang_match_file_content[pageid] = dst
        # 二次过滤：去掉目录索引
        for pageid, page in enumerate(lang_match_file_content):
            # dst = page
            dst = filter_index_title(file_index_titles, page)
            if dst:
                filtered_pages.append(dst)
    print(len(filtered_pages))
    # print(filtered_pages)
    return filtered_pages


BEGIN_TOKEN = re.compile(r'The meeting was called to order at') # 151
BEGIN_TOKEN2 = re.compile(r'The meeting (was )?resumed at') # 6
SUSPEND_TOKEN = re.compile(r'The meeting was suspended at')
ROSE_TOKEN = re.compile(r'The meeting rose at')
ROSE_TOKEN2 = re.compile(r'The discussion covered in the summary record ended at ')

SPEAKER_TOKEN = re.compile(r'^[A-Z].{2,25}( \(.*?\))?: ')


# INFO_PAGE_TOKEN = re.compile(r'United Nations\s.*?The meeting was called to order at ', flags=re.M | re.S)
INFO_PAGE_TOKEN = re.compile(r'United Nations\s.*?Corrections will be issued after the end of the session in a consolidated corrigendum\.', flags=re.M | re.S)
INFO_PAGE_TOKEN2 = re.compile(r'United Nations\s.*?Corrected records will be reissued electronically on the Official Document System of the United Nations \(http://documents\.un\.org\)\.', flags=re.M | re.S)
INFO_PAGE_TOKEN3 = re.compile(r'This record contains the text of speeches delivered in English.*?Corrected records will be reissued electronically on the Official Document System of the United Nations \(http://documents\.un\.org\)\.', flags=re.M | re.S)
