from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Union

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTChar, LTContainer, LTCurve, LTTextLine, TextLineElement
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser


@dataclass(frozen=True)
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1


class Page:
    def __init__(self, page: PDFPage, layout) -> None:
        _, _, width, height = page.mediabox
        self.width = width
        self.height = height

        rects = self.extract_rects(layout)
        self.frames = self.get_frames(rects)

        self.lines = self.aggregate_lt_text_lines(self.extract_lt_text_lines(layout))

    def extract_lt_text_lines(self, layout) -> list[LTTextLine]:
        if isinstance(layout, LTTextLine):
            return [layout]
        elif isinstance(layout, LTContainer):
            instances = []
            for child in layout:
                instances.extend(self.extract_lt_text_lines(child))
            return instances
        return []

    def aggregate_lt_text_lines(
        self,
        lt_text_lines: list[LTTextLine],
    ) -> list[dict[str, Union[str, list[LTChar], bool]]]:
        y2lt_text_lines, prev = defaultdict(list), 1000000
        for lt_text_line in sorted(lt_text_lines, key=lambda x: -x.y0):
            y = round(-lt_text_line.y0)
            if abs(prev - y) <= lt_text_line.height / 4:  # y軸上でoverlapしていたらまとめる
                popped = y2lt_text_lines.pop(prev)
                y2lt_text_lines[y].extend(popped)
            y2lt_text_lines[y].append(lt_text_line)
            prev = y

        aggregated = []
        for clustered in y2lt_text_lines.values():
            lt_chars = [
                text_line_element
                for text_line_element in chain.from_iterable(clustered)
                if self.is_valid_text_line_element(text_line_element)
            ]
            x2lt_char, prev = {}, 10000
            for lt_char in sorted(lt_chars, key=lambda x: x.x0):
                x0 = round(lt_char.x0)
                if abs(prev - x0) <= lt_char.width / 4:  # x軸上でoverlapしていたら無視
                    lt_char = x2lt_char.pop(prev)
                x2lt_char[x0] = lt_char
                prev = x0
            lt_chars = [*x2lt_char.values()]
            if len(lt_chars) >= 1:
                min_x0 = round(min(lc.x0 for lc in lt_chars))
                min_y0 = round(min(lc.y0 for lc in lt_chars))
                max_x1 = round(max(lc.x1 for lc in lt_chars))
                max_y1 = round(max(lc.y1 for lc in lt_chars))
                aggregated.append(
                    {
                        "lt_chars": lt_chars,
                        "table": any(
                            f.x0 - 1 <= min_x0 and f.y0 - 1 <= min_y0 and f.x1 + 1 >= max_x1 and f.y1 + 1 >= max_y1
                            for f in self.frames
                        ),
                        "line_break": max_x1 < self.width * 5 / 6,
                        "header": max_y1 >= self.height * 0.95,
                        "footer": min_y0 <= self.height * 0.05,
                    }
                )
        if aggregated:
            aggregated[-1]["line_break"] = False  # ページ末尾の行は右端に余白があっても改行しない
            for line in aggregated[1:-1]:
                # ページ先頭/末尾の行のみヘッダー/フッターとして扱う
                line["header"] = False
                line["footer"] = False
        return aggregated

    @staticmethod
    def is_valid_text_line_element(text_line_element: TextLineElement) -> bool:
        return (
            isinstance(text_line_element, LTChar)
            and len(text_line_element.get_text().strip()) == 1
            and text_line_element.x0 >= 0.0
            and text_line_element.y0 >= 0.0
        )

    def extract_rects(self, layout) -> list[Rect]:
        if isinstance(layout, LTCurve):
            try:
                rect = self.curve2rect(layout)
                if rect.x1 - rect.x0 < self.width * 5 / 6 or rect.y1 - rect.y0 < self.height * 5 / 6:
                    return [rect]
            except IndexError:
                pass
        elif isinstance(layout, LTContainer):
            instances = []
            for child in layout:
                instances.extend(self.extract_rects(child))
            return instances
        return []

    @staticmethod
    def curve2rect(curve: LTCurve) -> Rect:
        if curve is LTCurve:
            xs, ys = zip(*curve.pts)
            return Rect(x0=round(min(xs)), y0=round(min(ys)), x1=round(max(xs)), y1=round(max(ys)))
        else:
            return Rect(x0=round(curve.x0), y0=round(curve.y0), x1=round(curve.x1), y1=round(curve.y1))

    @staticmethod
    def get_frames(rects: list[Rect]) -> list[Rect]:
        y2xs = defaultdict(set)
        for rect in rects:
            for y in range(rect.y0, rect.y1 + 1):
                y2xs[y] |= {rect.x0, rect.x1}

        frames = []
        for rect in rects:
            if len(y2xs[round((rect.y0 + rect.y1) / 2)]) <= 2:
                continue
            frames.append(rect)
            overlapped = [
                f
                for f in frames
                if max(rect.x0, f.x0) <= min(rect.x1, f.x1) and max(rect.y0, f.y0) <= min(rect.y1, f.y1)
            ]
            frames = [f for f in frames if f not in overlapped]
            x0s, y0s, x1s, y1s = zip(*[f.to_tuple() for f in overlapped])
            frames.append(Rect(x0=min(x0s), y0=min(y0s), x1=max(x1s), y1=max(y1s)))
        return frames

    def to_text(
        self,
        include_table: bool = False,
        include_line_break: bool = False,
        include_header_and_footer: bool = False,
    ) -> str:
        text = ""
        for prev_line, cur_line, next_line in zip(
            self.lines[-1:] + self.lines[:-1], self.lines, self.lines[1:] + self.lines[:1]
        ):
            if cur_line["table"] is True:
                if include_table is True:
                    text += "\n" * int(prev_line["table"] is False)  # "<table>"
                    tr = ""  # "<tr>"
                    for cur_lt_char, next_lt_char in zip(
                        cur_line["lt_chars"], cur_line["lt_chars"][1:] + cur_line["lt_chars"][:1]
                    ):
                        tr += cur_lt_char.get_text().strip()
                        # 2文字以上離れていたら別カラムとみなす
                        tr += "\t" * int(next_lt_char.x0 - cur_lt_char.x1 >= cur_lt_char.width * 2.125)
                    tr += "\n"  # "</tr>"
                    text += tr
                    text += "\n" * int(next_line["table"] is False)  # "</table>"
            elif cur_line["header"] is True or cur_line["footer"] is True:
                if include_header_and_footer is True:
                    text += "".join(lc.get_text().strip() for lc in cur_line["lt_chars"])
                    text += "\u3000" * cur_line["line_break"] * include_line_break
            else:
                text += "".join(lc.get_text().strip() for lc in cur_line["lt_chars"])
                text += "\u3000" * cur_line["line_break"] * include_line_break
        return text


def extract_pages(in_file: Path) -> list[Page]:
    pages = []
    with in_file.open(mode="rb") as f:
        pdf_parser = PDFParser(f)
        pdf_document = PDFDocument(pdf_parser)
        pdf_resource_manager = PDFResourceManager()
        pdf_page_aggregator = PDFPageAggregator(pdf_resource_manager, laparams=LAParams(all_texts=True))
        pdf_page_interpreter = PDFPageInterpreter(pdf_resource_manager, pdf_page_aggregator)
        for page in PDFPage.create_pages(pdf_document):
            pdf_page_interpreter.process_page(page)
            layout = pdf_page_aggregator.get_result()
            pages.append(Page(page, layout))
    return pages
