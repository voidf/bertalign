import numpy as np

from bertalign import model
from bertalign.corelib import *
from bertalign.utils import *
lang_list = ["en", "zh", "fr", 'ru', "es"]
# lang_list = ["en", "zh"]

class Bertalign:
    def __init__(self, row, max_align=5, top_k=3, win=5, skip=-0.1, margin=True, len_penalty=True, is_splited=False, split_to_sents=False, log_func=print):
        self.max_align = max_align
        self.top_k = top_k
        self.win = win
        self.skip = skip
        self.margin = margin
        self.len_penalty = len_penalty
        self.log_func = log_func # 日志函数，会传入一个字符串
        self.record = row['record']

        sents = {}

        for lang in lang_list:
            sents[lang] = {}
            if split_to_sents:
                text_lines = []
                for s in row[lang].splitlines():
                    text_lines.extend(auto_split_sents(s, lang))
            else:
                if is_splited:
                    text_lines = row[lang].splitlines()
                else:
                    cleaned_text = clean_text(row[lang])
                    text_lines = split_sents(cleaned_text, lang)

            lines_length = len(text_lines)
            # special_lang = LANG.ISO[lang]
            log_func(f"record: {self.record}, lang: {lang}, sent len: {lines_length}")

            vecs, lens = model.transform(text_lines, max_align - 1)
            sents[lang]["lines_length"] = lines_length
            # sents[lang]["special_lang"] = special_lang
            sents[lang]["vecs"] = vecs
            sents[lang]["lens"] = lens
            sents[lang]["text_lines"] = text_lines

            if "en" in sents:
                sents[lang]["char_ratio"] = np.sum(lens[0,]) / np.sum(sents["en"]["lens"][0,])


        self.sents = sents
   
        

    def align_sents(self):

        result = {}
        benchmark_data = self.sents["en"]

        for lang in lang_list[1:]:
            src = self.sents[lang]
            # print("Performing first-step alignment ...") # 第一次对齐：原句对齐，所以只需要[0,:]
            D, I = find_top_k_sents(src["vecs"][0,:], benchmark_data["vecs"][0,:], k=self.top_k)
            first_alignment_types = get_alignment_types(2) 
            first_w, first_path = find_first_search_path(src["lines_length"], benchmark_data["lines_length"])
            first_pointers = first_pass_align(src["lines_length"], benchmark_data["lines_length"], first_w, first_path, first_alignment_types, D, I)
            first_alignment = first_back_track(src["lines_length"], benchmark_data["lines_length"], first_pointers, first_path, first_alignment_types)

            # print("Performing second-step alignment ...")
            second_alignment_types = get_alignment_types(self.max_align)
            second_w, second_path = find_second_search_path(first_alignment, self.win, src["lines_length"], benchmark_data["lines_length"])
            second_pointers = second_pass_align(src["vecs"], benchmark_data["vecs"], src["lens"], benchmark_data["lens"],
                                            second_w, second_path, second_alignment_types,
                                            src["char_ratio"], self.skip, margin=self.margin, len_penalty=self.len_penalty)
            second_alignment = second_back_track(src["lines_length"], benchmark_data["lines_length"], second_pointers, second_path, second_alignment_types)

            result[lang] = second_alignment

        self.result = result


    def create_result(self):
        result = {}
        for lang in self.result:
            result[lang] = ""
            for bead in (self.result[lang]):
                src_line = self._get_line(bead[0], self.sents[lang]["text_lines"])
                tgt_line = self._get_line(bead[1], self.sents["en"]["text_lines"])
                result[lang]+= src_line + "\n" + tgt_line + "\n \n"

        return result

    def print_sents(self):
        for lang in self.result:
            for bead in (self.result[lang]):
                src_line = self._get_line(bead[0], self.sents[lang]["text_lines"])
                tgt_line = self._get_line(bead[1], self.sents["en"]["text_lines"])
                print(src_line + "\n" + tgt_line + "\n")

            
    def yield_sents(self):
        for bead in (self.result):
            src_line = self._get_line(bead[0], self.src_sents)
            tgt_line = self._get_line(bead[1], self.tgt_sents)
            yield src_line + "\n++++++++++\n" + tgt_line + "\n"

    @staticmethod
    def _get_line(bead, lines):
        line = ''
        if len(bead) > 0:
            line = ' '.join(lines[bead[0]:bead[-1]+1])
        return line
