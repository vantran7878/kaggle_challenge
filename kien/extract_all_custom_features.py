import os
import re
import math
import ast
import numpy as np
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm

try:
    from magika import Magika
except ImportError:
    pass  # Sẽ báo lỗi cụ thể khi được gọi nếu thiếu biến

tqdm.pandas()

# ============================================================
# TỔNG HỢP LOGIC TRÍCH XUẤT ĐẶC TRƯNG TỪ CODE
# ============================================================

# Magika label → normalized language name
MAGIKA_LABEL_MAP = {
    'python':     'Python',
    'c':          'C',
    'cpp':        'C++',
    'java':       'Java',
    'cs':         'C#',
    'csharp':     'C#',
    'go':         'Go',
    'php':        'PHP',
    'javascript': 'JavaScript',
    'typescript': 'JavaScript',
    'shell':      'Python',
    'ruby':       'Python',
    'perl':       'Python',
}

# Keywords đa ngôn ngữ dùng để lọc identifier (giống preprocess_features.py)
MULTI_LANG_KEYWORDS = {
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'return',
    'int', 'float', 'double', 'char', 'void', 'bool', 'boolean', 'string', 'var', 'let', 'const',
    'def', 'function', 'class', 'struct', 'interface', 'package', 'import', 'using', 'namespace',
    'public', 'private', 'protected', 'static', 'final', 'const', 'try', 'catch', 'finally',
    'throw', 'throws', 'new', 'delete', 'true', 'false', 'null', 'nil', 'None', 'self', 'this',
    'func', 'defer', 'go', 'map', 'chan', 'type', 'range'
}

# Compiled regex dùng chung cho các agnostic features
_RE_WORDS      = re.compile(r'\w+')
_RE_CAMEL      = re.compile(r'[a-z][A-Z]')
_RE_DIGITS     = re.compile(r'\d')
_RE_EQ_SPACED  = re.compile(r' = ')
_RE_EQ_NOSPACE = re.compile(r'(?<=[^\s])=(?=[^\s])')

# ============================================================
# AGNOSTIC FEATURES (ag_1 → ag_10) — ported from preprocess_features.py
# ============================================================

def _analyze_identifiers(words: list) -> dict:
    """
    ag_1: id_len_avg      — Độ dài trung bình của identifier
    ag_2: id_entropy      — Entropy ký tự của tất cả identifier
    ag_3: id_short_ratio  — Tỉ lệ identifier ngắn (≤2 ký tự, vd: i, j, x)
    ag_4: id_num_ratio    — Tỉ lệ identifier chứa chữ số (vd: var1, temp2)
    """
    identifiers = [w for w in words if w not in MULTI_LANG_KEYWORDS and not w.isdigit()]

    if not identifiers:
        return {
            'id_len_avg':     0.0,
            'id_entropy':     0.0,
            'id_short_ratio': 0.0,
            'id_num_ratio':   0.0,
        }

    # ag_1 — Độ dài trung bình
    lens = [len(w) for w in identifiers]
    avg_len = float(np.mean(lens))

    # ag_2 — Entropy ký tự
    all_chars = "".join(identifiers)
    if not all_chars:
        entropy = 0.0
    else:
        char_counts = Counter(all_chars)
        total_chars = sum(char_counts.values())
        entropy = -sum(
            (c / total_chars) * math.log2(c / total_chars)
            for c in char_counts.values()
        )

    # ag_3 — Tỉ lệ identifier ngắn (≤2 ký tự)
    short_ratio = sum(1 for w in identifiers if len(w) <= 2) / len(identifiers)

    # ag_4 — Tỉ lệ identifier có chữ số
    num_ratio = sum(1 for w in identifiers if _RE_DIGITS.search(w)) / len(identifiers)

    return {
        'id_len_avg':     round(avg_len,     6),
        'id_entropy':     round(entropy,     6),
        'id_short_ratio': round(short_ratio, 6),
        'id_num_ratio':   round(num_ratio,   6),
    }


def _analyze_consistency(code: str, words: list) -> dict:
    """
    ag_5: style_consistency — Mức độ nhất quán giữa camelCase và snake_case
                              (1.0 = chỉ dùng 1 kiểu, 0.0 = trộn đều)
    ag_6: spacing_ratio     — Tỉ lệ dấu '=' không có khoảng trắng xung quanh
                              (LLM hay dùng ' = ', human hay bỏ sót khoảng trắng)
    """
    identifiers = [w for w in words if w not in MULTI_LANG_KEYWORDS]

    # ag_5 — Camel vs Snake consistency
    snake_count = sum(1 for w in identifiers if '_' in w)
    camel_count = sum(1 for w in identifiers if _RE_CAMEL.search(w))
    total_style = snake_count + camel_count

    if total_style == 0:
        consistency = 0.0
    else:
        consistency = abs(snake_count - camel_count) / total_style

    # ag_6 — Spacing quanh '='
    spaced   = len(_RE_EQ_SPACED.findall(code))
    nospaced = len(_RE_EQ_NOSPACE.findall(code))
    total_eq = spaced + nospaced

    spacing_ratio = (nospaced / total_eq) if total_eq > 0 else 0.0

    return {
        'style_consistency': round(consistency,   6),
        'spacing_ratio':     round(spacing_ratio, 6),
    }


def _analyze_structure(code: str, words: list) -> dict:
    """
    ag_7:  line_len_std  — Độ lệch chuẩn độ dài dòng (human → cao hơn)
    ag_8:  ttr           — Type-Token Ratio (đo sự lặp từ vựng)
    ag_9:  comment_ratio — Tỉ lệ dòng là comment
    ag_10: human_markers — Có TODO/FIXME/DEBUG/HACK hay không (0/1)
    """
    lines = code.split('\n')
    non_empty = [l for l in lines if l.strip()]

    # ag_7 — Std độ dài dòng
    if non_empty:
        line_lens = [len(l) for l in non_empty]
        line_std = float(np.std(line_lens))
    else:
        line_std = 0.0

    # ag_8 — TTR
    ttr = (len(set(words)) / len(words)) if words else 0.0

    # ag_9 — Comment ratio
    comment_lines = sum(
        1 for l in non_empty
        if l.strip().startswith(('#', '//', '/*'))
    )
    comment_ratio = comment_lines / (len(non_empty) + 1)

    # ag_10 — Human markers
    markers = len(re.findall(r'\b(TODO|FIXME|XXX|HACK|DEBUG)\b', code, re.IGNORECASE))
    human_markers = 1.0 if markers > 0 else 0.0

    return {
        'line_len_std':  round(line_std,      6),
        'ttr':           round(ttr,           6),
        'comment_ratio': round(comment_ratio, 6),
        'human_markers': human_markers,
    }


def extract_agnostic_features(code: str) -> dict:
    """
    Entry point cho 10 agnostic features (ag_1 → ag_10).
    Không cần model neural — chỉ dùng regex + thống kê.
    """
    if not isinstance(code, str):
        code = str(code)

    words = _RE_WORDS.findall(code)

    result = {}
    result.update(_analyze_identifiers(words))
    result.update(_analyze_consistency(code, words))
    result.update(_analyze_structure(code, words))
    return result


# ============================================================
# CÁC FEATURE GỐC (giữ nguyên)
# ============================================================

def get_llm_greeting_value(code: str) -> int:
    if not isinstance(code, str) or not code.strip():
        return 0
    LLM_GREETINGS = [
        r"here is", r"here's", r"certainly", r"sure,", r"sure thing",
        r"of course", r"let's break", r"as an ai", r"as a language model",
        r"i can help", r"below is", r"to solve this", r"hope this helps",
    ]
    pattern = r'(' + r'|'.join(LLM_GREETINGS) + r')'
    has_greeting = bool(re.search(pattern, code, re.IGNORECASE))
    has_backticks = '```' in code
    return 1 if (has_greeting or has_backticks) else 0

def extract_token_count(code: str) -> int:
    if not isinstance(code, str):
        return 0
    return len(re.findall(r'\w+', code))

def extract_maintainability_index(code_string: str) -> float:
    if not isinstance(code_string, str) or len(code_string) < 5:
        return 0.0
    lines = code_string.splitlines()
    loc = max(len(lines), 1)
    
    branch_count = len(re.findall(r'\b(if|elif|for|while|and|or|except|with|catch|switch|case)\b', code_string))
    cc = 1 + branch_count
    
    operators_pattern = r'(\+|-|\*|/|%|=|==|!=|>|<|>=|<=|&&|\|\||!|&|\||\^|~|<<|>>|\bif\b|\belse\b|\bfor\b|\bwhile\b|\breturn\b|\bdef\b|\bclass\b|\bimport\b)'
    operators_found = re.findall(operators_pattern, code_string)
    all_words = re.findall(r'\b[a-zA-Z0-9_]+\b', code_string)
    keywords_set = {'if', 'else', 'for', 'while', 'return', 'def', 'class', 'import', 'and', 'or', 'not'}
    operands_found = [w for w in all_words if w not in keywords_set]
    
    N = len(operators_found) + len(operands_found)
    eta = len(set(operators_found)) + len(set(operands_found))
    
    hv = N * math.log2(eta) if eta > 0 else 0.0
    
    ln_hv = math.log(hv) if hv > 0 else 0
    ln_loc = math.log(loc) if loc > 0 else 0
    
    mi_raw = 171 - 5.2 * ln_hv - 0.23 * cc - 16.2 * ln_loc
    return max(0.0, round(mi_raw * 100 / 171, 4))

def extract_internal_fan_out(code_string: str) -> float:
    if not isinstance(code_string, str) or len(code_string) < 5:
        return 0.0
    keywords_set = {'if', 'else', 'for', 'while', 'return', 'def', 'class', 'import', 'and', 'or', 'not'}
    function_calls = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_string)
    valid_calls = [c for c in function_calls if c not in keywords_set]
    return float(len(set(valid_calls)))

# Regex cho khai báo hàm đa ngôn ngữ
FUNC_DECL_PATTERN = re.compile(
    r'^\s*('
    r'def\s+\w+\s*\('              
    r'|func\s+\w+\s*\('            
    r'|function\s+\w+\s*\('        
    r'|\w[\w\s\*]+\s+\w+\s*\('    
    r'|public\s+\w+\s*\('          
    r'|private\s+\w+\s*\('
    r'|protected\s+\w+\s*\('
    r'|static\s+\w+\s+\w+\s*\('
    r')', re.MULTILINE
)

def _get_func_lengths_regex(code: str) -> list[int]:
    lines = code.splitlines()
    func_line_indices = [i for i, line in enumerate(lines) if FUNC_DECL_PATTERN.match(line)]
    if len(func_line_indices) < 2: return []
    lengths = [func_line_indices[i + 1] - func_line_indices[i] for i in range(len(func_line_indices) - 1)]
    lengths.append(len(lines) - func_line_indices[-1])
    return lengths

def extract_function_features(code_string: str) -> dict:
    if not isinstance(code_string, str) or len(code_string) < 10:
        return {'function_length_cv': 0.0, 'function_count': 0}
    lengths = []
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    lengths.append(node.end_lineno - node.lineno + 1)
    except Exception:
        pass

    if len(lengths) < 2:
        lengths = _get_func_lengths_regex(code_string)

    if len(lengths) < 2:
        return {'function_length_cv': 0.0, 'function_count': len(lengths)}

    arr = np.array(lengths, dtype=np.float32)
    mean = arr.mean()
    cv = arr.std() / (mean + 1e-9)
    return {'function_length_cv': round(float(cv), 4), 'function_count': len(lengths)}

TODO_PATTERN = re.compile(r'\b(TODO|FIXME|HACK|XXX|BUG|NOQA|TEMP|KLUDGE)\b', re.IGNORECASE)
PLACEHOLDER_PATTERN = re.compile(
    r'^\s*pass\s*$'                        
    r'|^\s*\.\.\.\s*$'                     
    r'|raise\s+NotImplementedError'        
    r'|throw\s+new\s+\w*Error\b'          
    r'|throw\s+new\s+\w*Exception\b'      
    r'|panic\s*\('                         
    r'|unimplemented!\s*\(\)',             
    re.MULTILINE
)

def extract_artifact_features(code_string: str) -> dict:
    if not isinstance(code_string, str) or len(code_string) < 5:
        return {'debug_artifact_score': 0.0, 'placeholder_ratio': 0.0}
    lines = code_string.splitlines()
    total_lines = max(len(lines), 1)
    todo_count = len(TODO_PATTERN.findall(code_string))
    placeholder_count = len(PLACEHOLDER_PATTERN.findall(code_string))
    placeholder_ratio = placeholder_count / total_lines
    debug_score = min((todo_count + placeholder_count * 2) / total_lines, 1.0)
    return {
        'debug_artifact_score': round(debug_score, 6),
        'placeholder_ratio': round(placeholder_ratio, 6)
    }

# ============================================================
# ENTRY POINT TỔNG HỢP
# ============================================================

def extract_all_custom_features(code: str) -> dict:
    """
    Trích xuất toàn bộ feature từ một đoạn code:
      - 4 features gốc (llm_greeting, token_count, maintainability_index, internal_fan_out)
      - 2 features hàm (function_length_cv, function_count)
      - 2 features artifact (debug_artifact_score, placeholder_ratio)
      - 10 agnostic features (ag_1→ag_10):
            id_len_avg, id_entropy, id_short_ratio, id_num_ratio,
            style_consistency, spacing_ratio,
            line_len_std, ttr, comment_ratio, human_markers
    """
    res = {
        'llm_greeting':          get_llm_greeting_value(code),
        'token_count':           extract_token_count(code),
        'maintainability_index': extract_maintainability_index(code),
        'internal_fan_out':      extract_internal_fan_out(code),
    }
    res.update(extract_function_features(code))
    res.update(extract_artifact_features(code))
    res.update(extract_agnostic_features(code))   # ag_1 → ag_10
    return res

# ============================================================
# MAIN PIPELINE 
# ============================================================

def add_features_to_dataframe(df: pd.DataFrame, code_col='code', detect_lang=False):
    if code_col not in df.columns:
        print(f"[!] DataFrame không chứa cột '{code_col}', không thể trích xuất feature!")
        return df

    if detect_lang:
        if 'language' in df.columns:
            print("[*] DataFrame đã có cột 'language', ghi đè lại bằng Magika...")
        else:
            print("[*] Bắt đầu nhận diện ngôn ngữ bằng Magika (chỉ áp dụng cho tập test)...")
        
        try:
            m = Magika()
            langs = []
            for code in tqdm(df[code_col], desc='  Detecting Lang', unit='sample'):
                try:
                    if isinstance(code, str):
                        code_bytes = code.encode('utf-8', errors='replace')
                    else:
                        code_bytes = code
                    result = m.identify_bytes(code_bytes)
                    magika_label = result.output.label
                    lang = MAGIKA_LABEL_MAP.get(magika_label, 'unknown')
                except Exception:
                    lang = 'unknown'
                langs.append(lang)
            df['language'] = langs
            print(f"  [+] Đã phân loại ngôn ngữ:\n{df['language'].value_counts().to_string()}\n")
        except NameError:
            print("[!] Có lỗi: Chưa cài đặt thư viện magika. Vui lòng chạy 'pip install magika'. Bỏ qua quá trình detect language.")

    print("[*] Bắt đầu trích xuất tuần tự (18 features từ code)...")
    extracted = df[code_col].progress_apply(extract_all_custom_features).apply(pd.Series)
    
    # Gộp vào DataFrame xoá trùng nếu có
    for col in extracted.columns:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    df = pd.concat([df, extracted], axis=1)
    return df

if __name__ == "__main__":
    # Test script - giả lập
    # data_path = "path/to/org_dataset.parquet"
    # df_org = pd.read_parquet(data_path)
    # is_test_file = "test" in data_path # Nếu là test file thì set detect_lang=True
    # df_modify = add_features_to_dataframe(df_org, detect_lang=is_test_file)
    # df_modify.to_parquet("path/to/output_dataset.parquet", index=False)
    print("Script đã sẵn sàng! Import add_features_to_dataframe(df, detect_lang=True/False) để dùng.")