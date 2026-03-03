import pytest
import re
# If the code is in your local fork, ensure the path is correct
from pymupdf4llm.helpers.utils import (
    _merge_single_letter_word_splits,
    _normalize_table_br_tags,
)

# Simulating the top-level function mentioned for integration tests
def normalize_table_text_mock(text):
    if not text:
        return text
    keep_leading = bool(re.match(r"^\s*<br\s*/?>", text, re.IGNORECASE))
    keep_trailing = bool(re.search(r"<br\s*/?>\s*$", text, re.IGNORECASE))
    # Simulates the table cleanup flow used in pymupdf_rag
    value = _normalize_table_br_tags(text)
    value = value.replace("\u00a0", " ").replace("\t", " ")
    value = _merge_single_letter_word_splits(value)
    value = re.sub(r"[ \t]+", " ", value)
    value = "\n".join(line.strip() for line in value.split("\n"))
    if keep_leading and value:
        value = f" {value}"
    if keep_trailing and value:
        value = f"{value} "
    return value

# List of test cases: (Input, Expected)
test_cases = [
    # --- Merge Cases (Single letter on one side) ---
    ("v  <br>ibration", "vibration"),        # Leading error with spaces
    ("lov <br>e", "love"),                  # Trailing error
    ("d <br>estination", "destination"),    # Leading error
    ("sourc <br>e", "source"),              # Trailing error
    ("A<br>pple", "Apple"),                 # Single initial letter
    ("Typ<br>o", "Typo"),                   # Single final letter
    ("p <br>rogramming", "programming"),    # With extra space
    ("pytho <br> n", "python"),             # With spaces on both sides
    ("I<br>nput", "Input"),
    ("Outpu<br>t", "Output"),
    ("F<br>ork", "Fork"),
    ("Tes<br>t", "Test"),
    ("R<br>ead", "Read"),
    ("Writ<br>e", "Write"),
    ("B<br>reak", "Break"),
    ("Fixe<br>d", "Fixed"),
    
    # --- Space Cases (More than one letter on both sides) ---
    ("Hello<br>World", "Hello World"),      # Full words
    ("First<br/>Second", "First Second"),   # Tag with slash
    ("Good<br >Bye", "Good Bye"),           # Tag with internal space
    ("Open<BR>Source", "Open Source"),      # Case insensitive
    ("Data<br>Base", "Data Base"),
    
    # --- Complex and Edge Cases ---
    ("1<br>23", "1 23"),                    # Numbers should not trigger merge (isalpha)
    ("A <br> B", "A B"),                    # Two single letters with spaces don't join
    ("<br>Starting", " Starting"),           # At the start of the string
    ("Ending<br>", "Ending "),               # At the end of the string
    ("Multiple <br> letters <br> h<br>ere", "Multiple letters here"), # Multiple
    ("Special @<br>char", "Special @ char"), # Special characters
    ("Word<br>  ", "Word "),                 # Empty spaces after
    ("   <br>Word", " Word"),                # Empty spaces before
    ("", ""),                               # Empty string
    (None, None),                           # None value
    
    # --- Tag Variations ---
    ("Small<br/>gap", "Small gap"),
    ("Big<br   />gap", "Big gap"),
    ("Mix<BR>Case", "Mix Case"),
    ("Final<br  >e", "Finale"),             # Isolated final 'e' letter
    # newline scenario that previously broke ASCII tables
    ("deformation v\nvibration", "deformation v\nvibration"),
]
@pytest.mark.parametrize("input_str, expected", test_cases, ids=[f"case_{i}" for i in range(len(test_cases))])
def test_normalize_table_br_tags(input_str, expected):
    """
    Testa a lógica principal de junção de palavras com log detalhado em caso de falha.
    """
    resultado = normalize_table_text_mock(input_str)
    
    # Mensagem customizada que será exibida apenas se o teste falhar
    error_message = (
        f"\nERRO NA NORMALIZAÇÃO:"
        f"\nEntrada:   '{input_str}'"
        f"\nEsperado:  '{expected}'"
        f"\nRecebido:  '{resultado}'"
    )
    
    assert resultado == expected, error_message

def test_integration_with_normalization_flow():
    """
    Testa se a normalização de espaços extras funciona junto com o br_tags.
    """
    raw_text = "  Normal <br> ization  test <br/>ing  "
    expected = "Normal ization test ing"
    result = normalize_table_text_mock(raw_text)
    
    assert result == expected, f"\nFalha na Integração:\nEsperado: '{expected}'\nRecebido: '{result}'"