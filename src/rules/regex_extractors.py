import re

DEFAULT_REGEX_LABELS = (
    "Номер телефона",
    "Сведения об ИНН",
    "Паспортные данные",
    "Номер банковского счета",
    "Номер карты",
    "Одноразовые коды",
    "Email",
)

PHONE_FORMATTED = re.compile(
    r"(?<!\w)(?:\+7[\s-]?\(?\d{3}\)?(?:[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}|[\s-]?\d{7})|7[\s-]?\(?\d{3}\)?[\s-]?\d{7}|8(?:[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}|-\d{3}-\d{3}-\d{2}-\d{2}|\d{10}))(?![\w:])"
)
PHONE_BARE10 = re.compile(r"(?<!\w)(?:4\d{9}|9\d{9})(?![\w:])")
PHONE_SHORT = re.compile(r"(?<!\w)(?:900|8800)(?!\w)")
EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9-])")
ACCOUNT = re.compile(r"(?<!\d)\d{20}(?!\d)")
CARD = re.compile(r"(?<![\w])(?:\d{16}|\d{4} \d{4} \d{4} \d{4})(?![\w=^])")
OTP_NUM = re.compile(r"(?<![A-Za-z0-9])\d{6}(?![A-Za-z0-9])")
INN_NUM = re.compile(r"(?<![\w])\d{10}(?:\d{2})?(?!\w)")
DATE_NUM = re.compile(r"(?<!\d)\d{2}\.\d{2}\.\d{4}(?!\d)")
DATE_TEXT = re.compile(r"(?<!\d)\d{1,2} [А-Яа-яё]+ \d{4} года")
DIV_CODE = re.compile(r"(?<!\d)\d{3}-\d{3}(?!\d)")
PASSPORT_COMBINED = re.compile(r"(?<!\d)(?:\d{4} \d{6}|\d{2} \d{2} \d{6})(?!\d)")
PASSPORT_SERIES = re.compile(r"\bсер(?:ия|ии|ией)\s*(?P<ent>\d{4}|\d{2} \d{2}|\d{2})\b", re.I)
PASSPORT_NUMBER = re.compile(r"\bномер(?:ом)?\s*(?P<ent>\d{6}|\d{7})\b", re.I)
PASSPORT_SERIA_NUM = re.compile(r"\bсер(?:ия|ии|ией)\s*(?P<s>\d{2})\s+(?P<n>\d{7})\b", re.I)
PASSPORT_GENERIC = re.compile(r"\bпаспортные данные\b(?=[^.!?\n]{0,40}у вас зарегистрированы)", re.I)
AUTH_START = re.compile(
    r"\b(?:Главным управлением|Отделом по|УФМС|ОУФМС|ФМС|ГУ МВД|МВД|УВД|ОВД|Отделом полиции|Отделом УФМС|Отделом МВД|Отделением по|Отдел УФМС|Управлением по вопросам миграции|Паспортным столом)\b"
)

INN_NEG_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"подтвердить актуальность инн",
        r"корректность введ[её]нного инн",
        r"верны ли данные по инн",
        r"новых тарифах для корпоративных клиентов",
        r"указан неверно в некоторых документах",
        r"выписк[ау] для организации с инн",
        r"выписк[ау] по сч[её]ту для организации с инн",
        r"проверки контрагента по инн",
        r"проверка партн[её]ров",
        r"информацию о текущих и новых тарифах",
        r"сменила юридический адрес",
        r"выписк[ау] по нашей организации",
        r"наша компания с инн .* должна обновить данные",
        r"данные по инн .* и огрн",
        r"при открытии нового счета для ооо",
        r"для обновления данных по инн .* и новым адресам",
        r"получить информацию о движении средств по инн .* через личный кабинет",
        r"можете подтвердить, что инн нашей организации",
        r"какой инн у компании",
        r"изменились реквизиты",
        r"проверить над[её]жность контрагента",
        r"для нашей компании с инн",
        r"нужен инн .* для нового договора",
        r"хотим получить информацию о движении средств",
        r"компания с инн .* числится",
        r"изменить почтовый адрес для компании",
        r"для обновления кпп .* по инн",
        r"поступление средств на сч[её]т .* для инн",
        r"закрыть сч[её]т для организации с инн",
        r"при открытии расч(?:е|ё)тного сч[её]та система не принимает инн",
        r"проверю статус вашего перевода по указанным реквизитам",
        r"инн .* остался прежним",
        r"не получила платеж на расч(?:е|ё)тный сч[её]т",
    ]
]
INN_POS_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"помимо инн",
        r"для аккредитива инн поставщика",
        r"является обязательным реквизитом для идентификации клиента",
        r"в реквизитах своего ип",
        r"в сбп по этому инн",
        r"смене реквизитов для компании",
        r"обычно отображается в разделе",
        r"обязательным реквизитом для открытия брокерского счета",
        r"оформления доверенности на сотрудника требуется инн нашей организации",
        r"движении средств по инн .* за прошлый квартал .* выписк",
        r"какой инн нашей организации нужно указать при оформлении новой корпоративной карты",
    ]
]

PHONE_CTX = (
    "телефон",
    "телефона",
    "телефону",
    "номеров",
    "мобильн",
    "оператор",
    "смс",
    "sms",
    "отправителя",
    "звоню",
    "звонок",
    "уведомлен",
    "привяз",
    "номер телефона",
)
PHONE_NUMBER_VERBS = (
    "удалить",
    "дополнительн",
    "домашн",
    "обслуживает",
    "приход",
    "получать",
    "получаю",
    "на номер",
    "с номера",
    "через сбп",
)
PHONE_EXCLUDE_CTX = ("водитель", "снилс", "паспорт", "инн")
NEG_ACCOUNT_RE = re.compile(r"расч(?:е|ё)тн\w*\s+сч", re.I)
ACCOUNT_NEG_MARKERS = (
    "указанным реквизитам",
    "для инн",
    "ожидаемого поступления",
    "поступление средств на счёт",
    "статус вашего перевода",
)


def validate_regex_labels(labels: list[str] | tuple[str, ...]) -> list[str]:
    unknown = sorted(set(labels) - set(DEFAULT_REGEX_LABELS))
    if unknown:
        raise ValueError(f"Unsupported regex labels: {unknown}. Supported labels: {list(DEFAULT_REGEX_LABELS)}")
    return list(dict.fromkeys(labels))


def extract_authorities(text: str) -> set[tuple[int, int]]:
    spans = []
    for match in AUTH_START.finditer(text):
        start = match.start()
        if any(existing_start <= start < existing_end for existing_start, existing_end in spans):
            continue

        end = start
        while end < len(text):
            if text.startswith(" с кодом подразделения", end):
                break
            char = text[end]
            if char == ",":
                break
            if char == ".":
                prev_char = text[end - 1] if end > 0 else ""
                if prev_char.lower() == "г":
                    end += 1
                    continue
                break
            end += 1

        entity = text[start:end].strip()
        if entity:
            spans.append((start, start + len(entity)))

    return set(spans)


def extract_phone(text: str) -> set[tuple[int, int]]:
    spans = {match.span() for match in PHONE_FORMATTED.finditer(text)}
    low = text.lower()

    for match in PHONE_BARE10.finditer(text):
        ctx = low[max(0, match.start() - 50): match.end() + 50]
        if any(token in ctx for token in PHONE_CTX) and not any(token in ctx for token in PHONE_EXCLUDE_CTX):
            spans.add(match.span())
            continue
        if "номер" in ctx and any(token in ctx for token in PHONE_NUMBER_VERBS) and not any(token in ctx for token in PHONE_EXCLUDE_CTX):
            spans.add(match.span())

    for match in PHONE_SHORT.finditer(text):
        ctx = low[max(0, match.start() - 35): match.end() + 35]
        if ("номер" in ctx or "номера" in ctx or "отправителя" in ctx or "с номера" in ctx) and not any(
            token in ctx for token in PHONE_EXCLUDE_CTX
        ):
            spans.add(match.span())

    return spans


def extract_inn(text: str) -> set[tuple[int, int]]:
    low = text.lower()
    if "инн" not in low:
        return set()
    if any(pattern.search(text) for pattern in INN_POS_PATTERNS):
        return {match.span() for match in INN_NUM.finditer(text)}
    if any(pattern.search(text) for pattern in INN_NEG_PATTERNS):
        return set()
    return {match.span() for match in INN_NUM.finditer(text)}


def extract_passport(text: str) -> set[tuple[int, int]]:
    low = text.lower()
    spans = set()

    if "паспорт" not in low and "загранпаспорт" not in low and "подразделения" not in low:
        return spans

    for match in PASSPORT_COMBINED.finditer(text):
        ctx = low[max(0, match.start() - 35): match.end() + 10]
        if ("паспорт" in ctx or "загранпаспорт" in ctx) and "водитель" not in ctx:
            spans.add(match.span())

    for match in PASSPORT_SERIA_NUM.finditer(text):
        spans.add(match.span("s"))
        spans.add(match.span("n"))

    for match in PASSPORT_SERIES.finditer(text):
        spans.add(match.span("ent"))

    for match in PASSPORT_NUMBER.finditer(text):
        spans.add(match.span("ent"))

    if "подразделения" in low:
        for match in DIV_CODE.finditer(text):
            spans.add(match.span())

    if "паспорт" in low or "загранпаспорт" in low:
        for match in DATE_NUM.finditer(text):
            ctx = low[max(0, match.start() - 45): match.end() + 30]
            if any(token in ctx for token in ["выдан", "срок действия", "истекает", "заканчивается"]):
                spans.add(match.span())

        for match in DATE_TEXT.finditer(text):
            ctx = low[max(0, match.start() - 45): match.end() + 30]
            if any(token in ctx for token in ["выдан", "срок действия", "истекает", "заканчивается"]):
                spans.add(match.span())

        spans |= extract_authorities(text)

    for match in PASSPORT_GENERIC.finditer(text):
        spans.add(match.span())

    return spans


def extract_account(text: str) -> set[tuple[int, int]]:
    low = text.lower()
    spans = set()
    for match in ACCOUNT.finditer(text):
        ctx = low[max(0, match.start() - 70): match.end() + 70]
        if "счет" not in ctx and "счёт" not in ctx:
            continue
        if NEG_ACCOUNT_RE.search(ctx):
            continue
        if any(token in ctx for token in ACCOUNT_NEG_MARKERS):
            continue
        spans.add(match.span())
    return spans


def extract_card(text: str) -> set[tuple[int, int]]:
    return {match.span() for match in CARD.finditer(text)}


def extract_otp(text: str) -> set[tuple[int, int]]:
    low = text.lower()
    if "код" not in low and "коды" not in low:
        return set()
    if "emv" in low:
        return set()
    if any(token in low for token in ["водитель", "паспорт", "загранпаспорт", "свидетельств", "инн", "огрн", "кпп", "снилс"]):
        return set()
    return {match.span() for match in OTP_NUM.finditer(text)}


def extract_email(text: str) -> set[tuple[int, int]]:
    return {match.span() for match in EMAIL.finditer(text)}


EXTRACTORS = {
    "Номер телефона": extract_phone,
    "Сведения об ИНН": extract_inn,
    "Паспортные данные": extract_passport,
    "Номер банковского счета": extract_account,
    "Номер карты": extract_card,
    "Одноразовые коды": extract_otp,
    "Email": extract_email,
}


def extract_regex_spans(text: str, enabled_labels: list[str] | tuple[str, ...]) -> list[tuple[int, int, str]]:
    labels = validate_regex_labels(enabled_labels)
    spans = []
    for label in labels:
        extractor = EXTRACTORS[label]
        spans.extend((start, end, label) for start, end in extractor(text))
    return sorted(set(spans), key=lambda item: (item[0], item[1], item[2]))
