from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets

try:
    from autogluon.tabular import TabularPredictor
except Exception:
    TabularPredictor = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "grape_variety_model"
KNOWLEDGE_BASE_CSV = BASE_DIR / "models" / "expert_grapes.csv"


FALLBACK_DATA = [
    ["Валек", "Белый (янтарный)", "12-15", "800-2000", "19-22"],
    ["Дубовский розовый", "Красно-розовый", "18-20", "700-800", "19-21"],
    ["Кардинал", "Красно-фиолетовый", "9-10", "600-800", "22-24"],
    ["Кодрянка", "Фиолетовый", "6-7", "400-600", "15-16"],
    ["Ливия", "Розовый", "10-12", "800-1000", "23"],
    ["Рошфор", "Черный", "7-9", "500-600", "19-21"],
    ["Сверхранний красный мускат", "Красный", "5-6", "350-400", "19-20"],
    ["Супер Экстра", "Белый (янтарный)", "10-12", "500-800", "24"],
    ["Тасон", "Бело-розовый", "7-8", "500-800", "19-20"],
    ["Юбилей Новочеркасска", "Розово-красный", "12-18", "700-1400", "18-19"],
    ["Аркадия (Настя)", "Белый", "7-9", "500-800", "16"],
    ["Белое чудо", "Белый", "6-8", "600-800", "19"],
    ["Гамлет", "Розовый", "14-15", "600-900", "21"],
    ["Дружба", "Белый (янтарный)", "4-6", "300-400", "19-21"],
    ["Жаворонок", "Белый", "4-5", "400-500", "19-20"],
    ["Монарх", "Белый (янтарный)", "12-14", "500-700", "18"],
    ["Надежда АЗОС", "Темно-синий", "8-11", "600-800", "17-18"],
    ["Пестрый", "Розовый", "8-10", "500-700", "20-25"],
    ["Светлана", "Белый", "12-16", "800-1000", "18-23"],
    ["Фрумоас Албэ", "Белый", "7-8", "400-600", "17-19"],
]

PROPERTY_DEFS = [
    {"ui": "Цвет ягоды", "kb": "Цвет", "ml": "Цвет", "type": "enum"},
    {"ui": "Размер ягоды", "kb": "Размер_ягоды", "ml": "Размер_ягоды", "type": "numeric"},
    {"ui": "Размер грозди", "kb": "Размер_грозди", "ml": "Размер_грозди", "type": "numeric"},
    {"ui": "Сахаристость", "kb": "Сахаристость", "ml": "Сахаристость", "type": "numeric"},
]

BASE_STYLES = """
QMainWindow, QWidget { background: #ffffff; color: #1b1e28; }
QLabel { color: #1b1e28; }
QListWidget { background: #ffffff; border: none; }
QComboBox {
    border: 2px solid #e3e6ee; border-radius: 6px; padding: 6px 10px;
    font-size: 14px; background: #ffffff;
}
QLineEdit {
    padding: 0 12px; border: 2px solid #e3e6ee; border-radius: 6px;
    font-size: 15px; color: #1b1e28; background: #ffffff;
}
QLineEdit:focus { border-color: #2b68ff; }
QToolButton { background: transparent; border: none; }
QToolButton:hover { text-decoration: underline; }
QPushButton.primary {
    background: #2b68ff; color: white; border: none;
    border-radius: 8px; font-size: 15px; font-weight: 600; padding: 10px 18px;
}
QPushButton.primary:hover { background: #2f74ff; }
QPushButton.primary:pressed { background: #275deb; }
QRadioButton, QCheckBox { font-size: 15px; padding-left: 2px; }
QRadioButton::indicator, QCheckBox::indicator { width: 16px; height: 16px; margin-right: 8px; }
QRadioButton::indicator:unchecked {
    border: 2px solid #b9c2d3; border-radius: 8px; background: #ffffff;
}
QRadioButton::indicator:checked {
    border: 2px solid #2b68ff; border-radius: 8px; background: #2b68ff;
}
QCheckBox::indicator:unchecked { border: 2px solid #b9c2d3; background: #ffffff; }
QCheckBox::indicator:checked { border: 2px solid #2b68ff; background: #2b68ff; }
"""


def parse_range(value: str):
    value = str(value).replace(" ", "").replace(",", ".")
    if "-" in value:
        left, right = value.split("-")
        return float(left), float(right)
    x = float(value)
    return x, x


def is_missing(value) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return value is None


def format_value(value):
    if is_missing(value):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


class GrapeClassifier:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.predictor = None
        self.error_message = ""
        self._load()

    def _load(self):
        if TabularPredictor is None:
            self.error_message = "Не установлен пакет autogluon."
            return

        if not self.model_dir.exists():
            self.error_message = f"Каталог модели не найден: {self.model_dir}"
            return

        try:
            self.predictor = TabularPredictor.load(str(self.model_dir))
        except Exception as exc:
            self.error_message = f"Не удалось загрузить модель: {exc}"

    @property
    def is_ready(self) -> bool:
        return self.predictor is not None

    def predict(self, features: dict, allowed_labels=None, top_n: int = 3):
        if not self.is_ready:
            raise RuntimeError(self.error_message or "Модель недоступна.")

        sample = pd.DataFrame(
            [features],
            columns=["Цвет", "Размер_ягоды", "Размер_грозди", "Сахаристость"]
        )

        proba = self.predictor.predict_proba(sample)

        if isinstance(proba, pd.DataFrame):
            scores = proba.iloc[0]
        else:
            raise RuntimeError("Не удалось получить вероятности классов от модели.")

        if allowed_labels is not None:
            scores = scores.reindex(allowed_labels).dropna()

        if scores.empty:
            raise RuntimeError("Модель не смогла оценить подходящие классы.")

        scores = scores.sort_values(ascending=False)
        predicted_label = scores.index[0]
        top_scores = scores.head(top_n)

        return predicted_label, top_scores


class KBModel:
    def __init__(self):
        self.properties = PROPERTY_DEFS
        self.properties_by_ui = {item["ui"]: item for item in self.properties}
        self.classifier = GrapeClassifier(MODEL_DIR)
        self.expert_df = self._load_expert_df()
        self.varieties = self.expert_df["Сорт_ягоды"].tolist()
        self.values = self._build_values()
        self.enum_options = self.expert_df["Цвет"].drop_duplicates().tolist()

    def _load_expert_df(self) -> pd.DataFrame:
        if KNOWLEDGE_BASE_CSV.exists():
            return pd.read_csv(KNOWLEDGE_BASE_CSV)
        return pd.DataFrame(
            FALLBACK_DATA,
            columns=["Сорт_ягоды", "Цвет", "Размер_ягоды", "Размер_грозди", "Сахаристость"]
        )

    def _build_values(self):
        result = {}
        for _, row in self.expert_df.iterrows():
            result[row["Сорт_ягоды"]] = {
                "Цвет": [row["Цвет"]],
                "Размер_ягоды": parse_range(row["Размер_ягоды"]),
                "Размер_грозди": parse_range(row["Размер_грозди"]),
                "Сахаристость": parse_range(row["Сахаристость"]),
            }
        return result

    def feature_row_from_inputs(self, inputs: dict) -> dict:
        row = {}
        for prop in self.properties:
            ui_name = prop["ui"]
            ml_name = prop["ml"]
            row[ml_name] = inputs.get(ui_name, np.nan)
        return row

    def _first_mismatch_reason(self, variety: str, inputs: dict):
        variety_values = self.values[variety]

        for prop in self.properties:
            ui_name = prop["ui"]
            kb_name = prop["kb"]
            prop_type = prop["type"]

            if ui_name not in inputs:
                continue

            user_value = inputs[ui_name]
            expert_value = variety_values[kb_name]

            if prop_type == "enum":
                expected = expert_value[0]
                if str(user_value).strip() != str(expected).strip():
                    return f"{ui_name}: {format_value(user_value)} вместо {expected}"

            elif prop_type == "numeric":
                lo, hi = expert_value
                try:
                    numeric_value = float(user_value)
                except (TypeError, ValueError):
                    return f"{ui_name}: некорректное значение"

                if not (lo <= numeric_value <= hi):
                    return f"{ui_name}: {format_value(numeric_value)} вне диапазона {format_value(lo)}–{format_value(hi)}"

        return None

    def expert_filter(self, inputs: dict):
        matched = []
        rejected = []

        for variety in self.varieties:
            reason = self._first_mismatch_reason(variety, inputs)
            if reason is None:
                matched.append(variety)
            else:
                rejected.append((variety, reason))

        return matched, rejected


class InputPage(QtWidgets.QWidget):
    go_to_kb = QtCore.pyqtSignal()

    def __init__(self, model: KBModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._current_inputs = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 12, 24, 16)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("Ввод исходных данных")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(24)
        layout.addLayout(row, 1)

        left = QtWidgets.QVBoxLayout()
        lbl_props = QtWidgets.QLabel("СВОЙСТВА")
        lbl_props.setStyleSheet("font-size:16px; font-weight:600;")
        left.addWidget(lbl_props)
        self.lwProps = QtWidgets.QListWidget()
        for item in self.model.properties:
            self.lwProps.addItem(item["ui"])
        left.addWidget(self.lwProps, 1)
        row.addLayout(left, 1)

        sep1 = QtWidgets.QFrame()
        sep1.setFixedWidth(1)
        sep1.setStyleSheet("background:#e9edf6;")
        row.addWidget(sep1)

        mid = QtWidgets.QVBoxLayout()
        lbl_val = QtWidgets.QLabel("ЗНАЧЕНИЕ")
        lbl_val.setStyleSheet("font-size:16px; font-weight:600;")
        mid.addWidget(lbl_val)

        self.stack = QtWidgets.QStackedLayout()
        self.num_panel = self._build_numeric_editor()
        self.enum_panel = self._build_enum_editor()
        self.stack.addWidget(self.num_panel)
        self.stack.addWidget(self.enum_panel)
        mid.addLayout(self.stack, 1)
        row.addLayout(mid, 1)

        sep2 = QtWidgets.QFrame()
        sep2.setFixedWidth(1)
        sep2.setStyleSheet("background:#e9edf6;")
        row.addWidget(sep2)

        right = QtWidgets.QVBoxLayout()
        lbl_sum = QtWidgets.QLabel("ИТОГО")
        lbl_sum.setStyleSheet("font-size:16px; font-weight:600;")
        right.addWidget(lbl_sum)
        self.sum_widget = QtWidgets.QListWidget()
        right.addWidget(self.sum_widget, 1)
        row.addLayout(right, 1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)

        self.btnViewKB = QtWidgets.QPushButton("Посмотреть базу знаний")
        self.btnViewKB.setProperty("class", "primary")
        self.btnViewKB.clicked.connect(self.go_to_kb.emit)

        self.btnDetect = QtWidgets.QPushButton("Определить сорт винограда")
        self.btnDetect.setProperty("class", "primary")
        self.btnDetect.clicked.connect(self._detect)

        btn_row.addWidget(self.btnViewKB)
        btn_row.addSpacing(12)
        btn_row.addWidget(self.btnDetect)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.lwProps.currentRowChanged.connect(self._on_prop_select)
        self.lwProps.setCurrentRow(0)
        self._refresh_summary()

    def _build_numeric_editor(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(16)

        lbl = QtWidgets.QLabel("Введите число")
        lbl.setStyleSheet("font-size: 16px;")
        self.edNumber = QtWidgets.QLineEdit()
        self.edNumber.setFixedWidth(140)
        self.edNumber.setPlaceholderText("0.0")
        self.edNumber.editingFinished.connect(self._on_number_entered)

        lay.addWidget(lbl)
        lay.addStretch(1)
        lay.addWidget(self.edNumber)
        lay.addStretch(3)
        return w

    def _build_enum_editor(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 8, 0, 0)
        v.setSpacing(10)

        self.enumGroup = QtWidgets.QButtonGroup(w)
        self.enumGroup.setExclusive(True)
        for opt in self.model.enum_options:
            rb = QtWidgets.QRadioButton(opt)
            self.enumGroup.addButton(rb)
            v.addWidget(rb)

        v.addStretch(1)
        self.enumGroup.buttonClicked.connect(lambda *_: self._on_enum_selected())
        return w

    def _on_prop_select(self, row: int):
        if row < 0:
            return

        ui_name = self.lwProps.item(row).text()
        prop = self.model.properties_by_ui[ui_name]
        prop_type = prop["type"]

        self.stack.setCurrentIndex(1 if prop_type == "enum" else 0)

        if prop_type == "numeric":
            val = self._current_inputs.get(ui_name)
            self.edNumber.setText("" if val is None else str(val))
        else:
            val = self._current_inputs.get(ui_name)
            for btn in self.enumGroup.buttons():
                btn.blockSignals(True)
                btn.setChecked(btn.text() == val)
                btn.blockSignals(False)

    def _on_number_entered(self):
        row = self.lwProps.currentRow()
        if row < 0:
            return

        ui_name = self.lwProps.item(row).text()
        text = self.edNumber.text().strip()

        if not text:
            self._current_inputs.pop(ui_name, None)
        else:
            try:
                self._current_inputs[ui_name] = float(text.replace(",", "."))
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Введите корректное число.")
                return

        self._refresh_summary()

    def _on_enum_selected(self):
        row = self.lwProps.currentRow()
        if row < 0:
            return

        ui_name = self.lwProps.item(row).text()
        checked = next((b.text() for b in self.enumGroup.buttons() if b.isChecked()), None)

        if checked:
            self._current_inputs[ui_name] = checked
        else:
            self._current_inputs.pop(ui_name, None)

        self._refresh_summary()

    def _refresh_summary(self):
        self.sum_widget.clear()
        for prop in self.model.properties:
            ui_name = prop["ui"]
            self.sum_widget.addItem(ui_name)
            value = self._current_inputs.get(ui_name)
            self.sum_widget.addItem(format_value(value))

    def _build_rejected_lines(self, rejected):
        if not rejected:
            return []

        lines = ["Почему остальные сорта не подходят:"]
        for variety, reason in rejected:
            lines.append(f"• {variety} – {reason}")
        return lines

    def _detect(self):
        if not self._current_inputs:
            QtWidgets.QMessageBox.information(
                self,
                "Недостаточно данных",
                "Введите хотя бы один признак для определения сорта."
            )
            return

        matched, rejected = self.model.expert_filter(self._current_inputs)

        if len(matched) == 0:
            lines = [
                "Экспертная система не нашла подходящего сорта."
            ]
            lines.extend(self._build_rejected_lines(rejected))

            QtWidgets.QMessageBox.information(
                self,
                "Результат",
                "\n".join(lines)
            )
            return

        if len(matched) == 1:
            lines = [
                f"Итоговый сорт: {matched[0]}",
                "Решение принято экспертной системой."
            ]
            lines.extend(self._build_rejected_lines(rejected))

            QtWidgets.QMessageBox.information(
                self,
                "Результат",
                "\n".join(lines)
            )
            return

        if not self.model.classifier.is_ready:
            lines = [
                "Экспертная система нашла несколько подходящих сортов:",
            ]
            for variety in matched:
                lines.append(f"• {variety}")

            lines.append("")
            lines.append("Модель машинного обучения недоступна, поэтому выбрать один сорт нельзя.")
            lines.extend(self._build_rejected_lines(rejected))

            QtWidgets.QMessageBox.warning(
                self,
                "Результат",
                "\n".join(lines)
            )
            return

        try:
            features = self.model.feature_row_from_inputs(self._current_inputs)
            predicted_label, top_scores = self.model.classifier.predict(
                features,
                allowed_labels=matched,
                top_n=min(3, len(matched))
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Ошибка", str(exc))
            return

        lines = [
            "Экспертная система нашла несколько подходящих сортов:",
        ]
        for variety in matched:
            lines.append(f"• {variety}")

        lines.append("")
        lines.append(f"Итоговый сорт: {predicted_label}")
        lines.append("Решение принято моделью машинного обучения среди подходящих сортов.")

        if not top_scores.empty:
            lines.append("")
            lines.append("Вероятности среди подходящих сортов:")
            for name, prob in top_scores.items():
                lines.append(f"• {name}: {prob * 100:.2f}%")

        lines.append("")
        lines.extend(self._build_rejected_lines(rejected))

        QtWidgets.QMessageBox.information(
            self,
            "Результат",
            "\n".join(lines)
        )


class KBViewPage(QtWidgets.QWidget):
    back_to_input = QtCore.pyqtSignal()

    def __init__(self, model: KBModel, parent=None):
        super().__init__(parent)
        self.model = model

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 12, 24, 16)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("Просмотр базы знаний")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        nav = QtWidgets.QHBoxLayout()
        self.btnPrev = QtWidgets.QToolButton()
        self.btnNext = QtWidgets.QToolButton()
        for b in (self.btnPrev, self.btnNext):
            b.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            b.setStyleSheet("QToolButton{color:#2b68ff; font-weight:600;}")

        self.cbVariety = QtWidgets.QComboBox()
        self.cbVariety.addItems(self.model.varieties)
        self.cbVariety.setFixedWidth(260)

        nav.addWidget(self.btnPrev)
        nav.addStretch(1)
        nav.addWidget(self.cbVariety)
        nav.addStretch(1)
        nav.addWidget(self.btnNext)
        layout.addLayout(nav)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(24)

        left = QtWidgets.QVBoxLayout()
        cap = QtWidgets.QLabel("СВОЙСТВА")
        cap.setStyleSheet("font-size:16px; font-weight:600;")
        left.addWidget(cap)
        self.lwProps = QtWidgets.QListWidget()
        for item in self.model.properties:
            self.lwProps.addItem(item["ui"])
        left.addWidget(self.lwProps, 1)
        row.addLayout(left, 1)

        sep = QtWidgets.QFrame()
        sep.setFixedWidth(1)
        sep.setStyleSheet("background:#e9edf6;")
        row.addWidget(sep)

        right = QtWidgets.QVBoxLayout()
        cap2 = QtWidgets.QLabel("ЗНАЧЕНИЯ")
        cap2.setStyleSheet("font-size:16px; font-weight:600;")
        right.addWidget(cap2)

        self.valuesLayout = QtWidgets.QHBoxLayout()
        self.valuesLayout.setSpacing(30)

        self.lblFromCap = QtWidgets.QLabel("От")
        self.lblFromVal = QtWidgets.QLabel("")
        self.lblToCap = QtWidgets.QLabel("До")
        self.lblToVal = QtWidgets.QLabel("")

        self.valuesLayout.addWidget(self.lblFromCap)
        self.valuesLayout.addWidget(self.lblFromVal)
        self.valuesLayout.addSpacing(10)
        self.valuesLayout.addWidget(self.lblToCap)
        self.valuesLayout.addWidget(self.lblToVal)
        self.valuesLayout.addStretch(1)

        right.addLayout(self.valuesLayout)
        row.addLayout(right, 2)
        layout.addLayout(row, 1)

        btnRow = QtWidgets.QHBoxLayout()
        btnRow.addStretch(1)
        self.btnBack = QtWidgets.QPushButton("Вернуться к вводу исходных данных")
        self.btnBack.setProperty("class", "primary")
        btnRow.addWidget(self.btnBack)
        btnRow.addStretch(1)
        layout.addLayout(btnRow)

        self.cbVariety.currentIndexChanged.connect(self._refresh)
        self.btnPrev.clicked.connect(lambda: self._shift(-1))
        self.btnNext.clicked.connect(lambda: self._shift(1))
        self.lwProps.currentRowChanged.connect(self._refresh)
        self.btnBack.clicked.connect(self.back_to_input.emit)

        self.cbVariety.setCurrentIndex(0)
        self.lwProps.setCurrentRow(0)
        self._refresh()

    def _shift(self, delta: int):
        n = len(self.model.varieties)
        i = (self.cbVariety.currentIndex() + delta) % n
        self.cbVariety.setCurrentIndex(i)

    def _refresh(self):
        current_index = self.cbVariety.currentIndex()
        variety = self.model.varieties[current_index]

        prev_i = (current_index - 1) % len(self.model.varieties)
        next_i = (current_index + 1) % len(self.model.varieties)
        self.btnPrev.setText(f"← {self.model.varieties[prev_i]}")
        self.btnNext.setText(f"{self.model.varieties[next_i]} →")

        current_item = self.lwProps.currentItem()
        if current_item is None:
            return

        ui_name = current_item.text()
        prop = self.model.properties_by_ui[ui_name]
        kb_name = prop["kb"]
        prop_type = prop["type"]
        data = self.model.values.get(variety, {}).get(kb_name)

        if prop_type == "numeric":
            rng = data if data else ("—", "—")
            self.lblFromCap.setText("От")
            self.lblFromVal.setText(format_value(rng[0]))
            self.lblToCap.setText("До")
            self.lblToVal.setText(format_value(rng[1]))
            self.lblToCap.show()
            self.lblToVal.show()
        else:
            self.lblFromCap.setText("")
            self.lblFromVal.setText(", ".join(data or []))
            self.lblToCap.hide()
            self.lblToVal.hide()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Определение сорта винограда")
        self.resize(1080, 700)
        self.setStyleSheet(BASE_STYLES)

        self.model = KBModel()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        self.stackHost = QtWidgets.QStackedLayout(central)

        self.pageInput = InputPage(self.model)
        self.pageKB = KBViewPage(self.model)

        self.stackHost.addWidget(self.pageInput)
        self.stackHost.addWidget(self.pageKB)

        self.pageInput.go_to_kb.connect(lambda: self.stackHost.setCurrentIndex(1))
        self.pageKB.back_to_input.connect(lambda: self.stackHost.setCurrentIndex(0))


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()