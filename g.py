import datetime
import os

from pathlib import Path
from datasets import load_dataset
from bertalign import Bertalign
import process_zh_text
import process_en_text
from helper import dump_align_result_to_file



LANGS = ['zh', 'fr', 'es', 'ru']



def cat(*args): return '/'.join(args)



def without_preprocess(row):
    aligner = Bertalign(row)
    aligner.align_sents()
    result = aligner.create_result()
    dump_align_result_to_file(row['record'], result)


 # dst = 'en'
    # dst_text = row[dst].replace('\n----\n', '\n')
    # # zh = row['zh'].replace('\n----\n', '\n')
    # for src in langs:
    #     src_text = row[src].replace('\n----\n', '\n')
    #     output_dir = f'./aligned/{src}_{dst}/'
    #     Path(output_dir).mkdir(parents=True, exist_ok=True)
    #     output_filename = f'{output_dir}{id}.txt'

    #     if os.path.exists(output_filename):
    #         print('skip', output_filename)
    #         continue

    #     aligner = Bertalign(src_text, dst_text, src_lang=src, tgt_lang=dst)
    #     aligner.align_sents()

    #     with open(f'{output_filename}', 'w', encoding='utf-8') as f:
    #         for aligned in aligner.yield_sents():
    #             f.write(aligned + '=' * 10 + '\n')   

def dump_src(row: str):
    rec = row['record']
    for lang in ['zh', 'en', 'fr', 'es', 'ru']:
        with open(f"./pre/{lang}src.txt", 'a', encoding='utf-8') as f:
            f.write(f'''
==========
{rec}
==========
'''         + row[lang])
    

def with_preprocess(row: str):
    Path("./done").mkdir(parents=True, exist_ok=True)
    rec = row['record']

    # zh_text = process_zh_text.start(row["zh"])
    # en_text = process_en_text.start(row["en"])





#     row["zh"] = zh_text
#     row["en"] = en_text
#     with open(f"./pre/zhall.txt", 'a', encoding='utf-8') as f:
#         f.write(f'''
# ==========
# {rec}
# ==========
# '''         + zh_text)
#     with open(f"./pre/enall.txt", 'a', encoding='utf-8') as f:
#         f.write(f'''
# ==========
# {rec}
# ==========
# '''         + en_text)

    # with open(f"./pre/{rec}zh.txt", "w", encoding='utf-8') as f:
    #     f.write(zh_text)

    # with open(f"./pre/{rec}en.txt", "w", encoding='utf-8') as f:
    #     f.write(en_text)

    # without_preprocess(row)






if __name__ == "__main__":
    # os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    Path(f"./pre").mkdir(parents=True, exist_ok=True)
    begin_time = datetime.datetime.now()
    dataset = load_dataset("ranWang/UN_Historical_PDF_Article_Text_Corpus", split='randomTest')
    dataset.map(with_preprocess)

    # print(process_zh_text.start(dataset[0]["zh"]))

    end_time = datetime.datetime.now()
    print('Time elapsed:', end_time - begin_time)
