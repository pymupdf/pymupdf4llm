# PyMuPDF4LLM - Guia de Uso em Projetos

Este é um fork do PyMuPDF4LLM com funcionalidades estendidas para extração de tabelas, incluindo representação ASCII de tabelas com células mescladas.

## Instalação

### Opção 1: Instalação Local (Desenvolvimento)

Se você está desenvolvendo ou usando este fork localmente:

```bash
# Clone o repositório
git clone <url-do-repositorio>
cd pymupdf4llm

# Instale em modo desenvolvimento
cd pymupdf4llm
pip install -e .
```

### Opção 2: Instalação como Pacote

Se você quer usar este pacote em outro projeto:

```bash
# No diretório do projeto pymupdf4llm
cd pymupdf4llm
pip install -e .

# Ou instale diretamente do diretório
pip install /caminho/para/pymupdf4llm/pymupdf4llm
```

### Opção 3: Adicionar como Dependência em requirements.txt

```txt
# requirements.txt
-e /caminho/para/pymupdf4llm/pymupdf4llm
# ou
pymupdf4llm @ file:///caminho/para/pymupdf4llm/pymupdf4llm
```

## Uso Básico

### Importação

```python
import pymupdf4llm
# ou
import pymupdf4llm as llm
```

### Converter PDF para Markdown (Texto Simples)

```python
import pymupdf4llm

# Converter todo o PDF para uma string Markdown
md_text = pymupdf4llm.to_markdown("documento.pdf")

# Salvar em arquivo
with open("output.md", "w", encoding="utf-8") as f:
    f.write(md_text)
```

### Processar Páginas Específicas

```python
# Processar apenas as páginas 0, 2 e 5 (índices baseados em 0)
md_text = pymupdf4llm.to_markdown("documento.pdf", pages=[0, 2, 5])
```

## Extração de Tabelas com Estrutura Detalhada

### Uso Básico com `page_chunks=True`

Para obter informações detalhadas sobre tabelas, use `page_chunks=True`:

```python
import pymupdf4llm

# Processar PDF com informações estruturadas
chunks = pymupdf4llm.to_markdown("documento.pdf", page_chunks=True)

# chunks é uma lista de dicionários, um para cada página
for idx, chunk in enumerate(chunks):
    print(f"Página {idx + 1}:")
    print(f"  - {len(chunk.get('tables', []))} tabela(s)")
    print(f"  - {len(chunk.get('images', []))} imagem(ns)")
```

### Estrutura de Retorno

Cada chunk (página) contém:

```python
{
    "metadata": {
        "file_path": "caminho/do/arquivo.pdf",
        "page_count": 10,
        "page": 1,  # número da página (1-based)
        # ... outros metadados do PDF
    },
    "toc_items": [...],  # Ítens do índice
    "tables": [...],     # Lista de tabelas (ver abaixo)
    "images": [...],     # Lista de imagens
    "graphics": [...],   # Gráficos vetoriais
    "text": "...",       # Texto em Markdown
    "words": [...]       # Palavras individuais (se extract_words=True)
}
```

## Estrutura das Tabelas

### Acessando Tabelas

```python
chunks = pymupdf4llm.to_markdown("documento.pdf", page_chunks=True)

for chunk in chunks:
    tabelas = chunk.get("tables", [])
    
    for tabela in tabelas:
        print(f"Bbox: {tabela['bbox']}")
        print(f"Linhas: {tabela['rows']}")
        print(f"Colunas: {tabela['columns']}")
        print(f"Markdown: {tabela['markdown']}")
        print(f"Matriz: {tabela['matriz']}")
        print(f"Matriz ASCII: {tabela['matriz_ascii']}")  # NOVO!
```

### Estrutura Completa de uma Tabela

```python
{
    "bbox": (x0, y0, x1, y1),  # Coordenadas da tabela
    "rows": 5,                  # Número de linhas
    "columns": 3,               # Número de colunas
    "matriz": [                 # Matriz de células com metadados
        [
            {
                "text": "Conteúdo da célula",
                "row": 0,
                "col": 0,
                "rowspan": 1,
                "colspan": 1,
                "bbox": (x0, y0, x1, y1),
                "is_merged": False,
                "merged_from": None
            },
            # ... mais células
        ],
        # ... mais linhas
    ],
    "markdown": "| Coluna 1 | Coluna 2 |\n| --- | --- |\n| ...",
    "matriz_ascii": "----------------------\nHeader |\n----------------------\nCell 1 | Cell 2 |\n..."
}
```

### Campo `matriz_ascii` (NOVO)

O campo `matriz_ascii` contém uma representação ASCII da tabela que preserva células mescladas:

```python
chunks = pymupdf4llm.to_markdown("documento.pdf", page_chunks=True)

for chunk in chunks:
    for tabela in chunk.get("tables", []):
        print("Tabela em formato ASCII:")
        print(tabela["matriz_ascii"])
```

**Exemplo de saída:**

```
----------------------
Header |
----------------------
Cell 1 | Cell 2 |
----------------------
Cell 3 | Cell 4 |
----------------------
```

## Estrutura de uma Célula

Cada célula na matriz contém:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `text` | `str` | Texto contido na célula |
| `row` | `int` | Índice da linha (0-based) |
| `col` | `int` | Índice da coluna (0-based) |
| `rowspan` | `int` | Número de linhas que a célula ocupa |
| `colspan` | `int` | Número de colunas que a célula ocupa |
| `bbox` | `tuple` ou `None` | Coordenadas `(x0, y0, x1, y1)` |
| `is_merged` | `bool` | Se é uma célula mesclada secundária |
| `merged_from` | `tuple` ou `None` | Posição `(row, col)` da célula primária |

