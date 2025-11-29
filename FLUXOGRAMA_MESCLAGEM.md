# Fluxograma: Detecção de Células Mescladas e Montagem da Matriz ASCII

## Parte 1: Detecção de Células Mescladas (is_merged)

```
┌─────────────────────────────────────────────────────────────┐
│ INÍCIO: Processar Tabela do PDF                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │ Coletar todas as células não vazias   │
        │ e criar cell_rects_map                │
        │ (row, col) → bbox                     │
        └───────────────────────┬───────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │ Para cada célula na tabela:           │
        │ ┌───────────────────────────────────┐ │
        │ │ Extrair texto da célula           │ │
        │ │ Extrair bbox (coordenadas)        │ │
        │ └───────────────────────────────────┘ │
        └───────────────────────┬───────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │ Calcular razões de tamanho:           │
        │ height_ratio = cell.height / avg_height│
        │ width_ratio = cell.width / avg_width  │
        └───────────────────────┬───────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │ Verificar se é maior que 1.3x:        │
        │                                       │
        │ height_ratio > 1.3?                   │
        │   └─► SIM → rowspan = round(height_ratio)│
        │   └─► NÃO → rowspan = 1               │
        │                                       │
        │ width_ratio > 1.3?                    │
        │   └─► SIM → colspan = round(width_ratio)│
        │   └─► NÃO → colspan = 1               │
        └───────────────────────┬───────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │ Limitar valores:                       │
        │ rowspan = min(rowspan, rows - row_idx)│
        │ colspan = min(colspan, cols - col_idx)│
        └───────────────────────┬───────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │ VERIFICAÇÃO DE SOBREPOSIÇÃO           │
        │ (Validar se realmente é mesclada)     │
        └───────────────────────┬───────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
    ┌───────────────────────┐      ┌───────────────────────┐
    │ Se rowspan > 1:       │      │ Se colspan > 1:       │
    │                       │      │                       │
    │ Para cada linha       │      │ Para cada coluna      │
    │ coberta:              │      │ coberta:              │
    │                       │      │                       │
    │ Verificar se existe   │      │ Verificar se existe   │
    │ célula física em      │      │ célula física em      │
    │ (check_row, col_idx)  │      │ (row_idx, check_col)  │
    └───────────┬───────────┘      └───────────┬───────────┘
                │                               │
                ▼                               ▼
    ┌───────────────────────┐      ┌───────────────────────┐
    │ Calcular overlap_ratio│      │ Calcular overlap_ratio│
    │ = área sobreposta /   │      │ = área sobreposta /   │
    │   menor área          │      │   menor área          │
    └───────────┬───────────┘      └───────────┬───────────┘
                │                               │
                ▼                               ▼
    ┌───────────────────────┐      ┌───────────────────────┐
    │ overlap_ratio < 0.5?  │      │ overlap_ratio < 0.5?  │
    │ (menos de 50%)        │      │ (menos de 50%)        │
    └───────────┬───────────┘      └───────────┬───────────┘
                │                               │
        ┌───────┴───────┐               ┌───────┴───────┐
        │               │               │               │
        ▼               ▼               ▼               ▼
    ┌───────┐      ┌──────────┐   ┌───────┐      ┌──────────┐
    │ SIM   │      │ NÃO      │   │ SIM   │      │ NÃO      │
    │       │      │          │   │       │      │          │
    │ Célula│      │ Reduzir  │   │ Célula│      │ Reduzir  │
    │ separada│    │ rowspan  │   │ separada│    │ colspan  │
    │ encontrada│  │          │   │ encontrada│  │          │
    └───────┘      └──────────┘   └───────┘      └──────────┘
        │               │               │               │
        └───────────────┴───────────────┴───────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │ Criar célula primária:                │
        │ {                                     │
        │   "text": "...",                      │
        │   "row": row_idx,                     │
        │   "col": col_idx,                     │
        │   "rowspan": rowspan,                 │
        │   "colspan": colspan,                 │
        │   "is_merged": False,  ← PRIMÁRIA     │
        │   "merged_from": None                 │
        │ }                                     │
        └───────────────────────┬───────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │ Se rowspan > 1 OU colspan > 1:        │
        │                                       │
        │ Para cada posição coberta:            │
        │   (row_idx + r_offset,                │
        │    col_idx + c_offset)                │
        │                                       │
        │ Criar célula secundária:              │
        │ {                                     │
        │   "text": "...", (mesmo texto)        │
        │   "row": covered_row,                 │
        │   "col": covered_col,                 │
        │   "rowspan": 1,                       │
        │   "colspan": 1,                       │
        │   "is_merged": True,  ← SECUNDÁRIA    │
        │   "merged_from": (row_idx, col_idx)   │
        │ }                                     │
        └───────────────────────┬───────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │ Matriz completa com células marcadas  │
        │ - Primárias: is_merged=False          │
        │ - Secundárias: is_merged=True         │
        └───────────────────────────────────────┘
```

## Parte 2: Montagem da Matriz ASCII

```
┌─────────────────────────────────────────────────────────────┐
│ INÍCIO: matriz_to_ascii(matrix)                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │ FASE 1: Preparação dos Dados          │
        │                                       │
        │ Para cada linha na matriz:            │
        │   Para cada célula na linha:          │
        │     ┌─────────────────────────────┐   │
        │     │ Se is_merged=True:          │   │
        │     │   text = "" (vazio)         │   │
        │     │ Senão:                      │   │
        │     │   text = cell["text"]       │   │
        │     └─────────────────────────────┘   │
        │                                       │
        │ Guardar:                              │
        │ - row_cells: células originais        │
        │ - row_texts: apenas textos            │
        │ - max_cols: número máximo de colunas  │
        └───────────────────────┬───────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │ FASE 2: Calcular Larguras das Colunas │
        │                                       │
        │ Para cada coluna:                     │
        │   Para cada linha:                    │
        │     largura = max(largura, len(texto))│
        │                                       │
        │ Resultado: col_widths[]               │
        │ Exemplo: [6, 6, 8]                    │
        └───────────────────────┬───────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │ FASE 3: Construir Linha de Conteúdo   │
        │ build_content_line(row_index)         │
        └───────────────────────┬───────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │ Para cada coluna (col = 0 até max_cols):│
        │                                       │
        │ ┌─────────────────────────────────┐  │
        │ │ Obter célula atual              │  │
        │ └─────────────────────────────────┘  │
        │                                       │
        │ ┌─────────────────────────────────┐  │
        │ │ É célula mesclada secundária?   │  │
        │ │ (is_merged=True)                │  │
        │ └───────────┬─────────────────────┘  │
        │             │                         │
        │     ┌───────┴───────┐                │
        │     │               │                 │
        │     ▼               ▼                 │
        │ ┌─────────┐   ┌──────────────────┐   │
        │ │ SIM     │   │ NÃO              │   │
        │ │         │   │                  │   │
        │ │ Criar   │   │ Obter colspan    │   │
        │ │ espaço  │   │                  │   │
        │ │ vazio:  │   │ ┌──────────────┐ │   │
        │ │ "".ljust│   │ │ colspan == 1?│ │   │
        │ │ (width) │   │ └──────┬───────┘ │   │
        │ │ + " |"  │   │        │         │   │
        │ └─────────┘   │   ┌─────┴─────┐   │   │
        │               │   │           │   │   │
        │               │   ▼           ▼   │   │
        │               │ ┌─────┐   ┌──────────┐│
        │               │ │ SIM │   │ NÃO      ││
        │               │ │     │   │          ││
        │               │ │ Célula│ │ Célula   ││
        │               │ │ normal│ │ mesclada ││
        │               │ │      │   │          ││
        │               │ │ text.│   │ Calcular ││
        │               │ │ ljust│   │ largura  ││
        │               │ │ (width)│ │ total:   ││
        │               │ │ + " |"│ │ sum(widths)│
        │               │ └─────┘   │ + espaços ││
        │               │           │          ││
        │               │           │ text.ljust││
        │               │           │ (total)  ││
        │               │           │ + " |"   ││
        │               │           └──────────┘│
        │               └───────────────────────┘
        │                                       │
        │ Avançar: col += colspan               │
        └───────────────────────┬───────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │ FASE 4: Montar Saída Final            │
        │                                       │
        │ 1. Construir primeira linha            │
        │ 2. Calcular comprimento do separador   │
        │    separator = "-" * len(primeira_linha)│
        │                                       │
        │ 3. Para cada linha:                    │
        │    output.append(separator)           │
        │    output.append(conteudo)            │
        │                                       │
        │ 4. Adicionar separador final          │
        │                                       │
        │ 5. Retornar: "\n".join(output)       │
        └───────────────────────────────────────┘
```

## Exemplo Prático Completo

### Entrada (Matriz):
```python
[
    [
        {"text": "Header", "row": 0, "col": 0, "colspan": 2, "is_merged": False},
        {"text": "Header", "row": 0, "col": 1, "is_merged": True, "merged_from": (0,0)}
    ],
    [
        {"text": "Cell 1", "row": 1, "col": 0, "colspan": 1, "is_merged": False},
        {"text": "Cell 2", "row": 1, "col": 1, "colspan": 1, "is_merged": False}
    ]
]
```

### Processamento:

**Fase 1 - Preparação:**
```
row_cells[0] = [{"text": "Header", ...}, {"text": "Header", "is_merged": True, ...}]
row_texts[0] = ["Header", ""]  ← Célula secundária vazia
row_cells[1] = [{"text": "Cell 1", ...}, {"text": "Cell 2", ...}]
row_texts[1] = ["Cell 1", "Cell 2"]
max_cols = 2
```

**Fase 2 - Larguras:**
```
col_widths[0] = max(len("Header"), len("Cell 1")) = max(6, 6) = 6
col_widths[1] = max(len(""), len("Cell 2")) = max(0, 6) = 6
```

**Fase 3 - Construir Linha 0:**
```
col = 0:
  - Célula: {"text": "Header", "colspan": 2, "is_merged": False}
  - colspan = 2
  - total_width = 6 + 6 + (2-1)*3 = 15
  - segment = "Header".ljust(15) + " |" = "Header         |"
  - col += 2 (pula coluna 1)

Resultado linha 0: "Header         |"
```

**Fase 3 - Construir Linha 1:**
```
col = 0:
  - Célula: {"text": "Cell 1", "colspan": 1}
  - segment = "Cell 1".ljust(6) + " |" = "Cell 1 |"
  - col += 1

col = 1:
  - Célula: {"text": "Cell 2", "colspan": 1}
  - segment = "Cell 2".ljust(6) + " |" = "Cell 2 |"
  - col += 1

Resultado linha 1: "Cell 1 | Cell 2 |"
```

**Fase 4 - Montagem Final:**
```
separator = "-" * len("Header         |") = "----------------------"

output = [
    "----------------------",
    "Header         |",
    "----------------------",
    "Cell 1 | Cell 2 |",
    "----------------------"
]

Resultado: "----------------------\nHeader         |\n----------------------\nCell 1 | Cell 2 |\n----------------------"
```

### Saída Final:
```
----------------------
Header         |
----------------------
Cell 1 | Cell 2 |
----------------------
```

## Pontos-Chave

1. **Detecção de Mesclagem:**
   - Compara tamanho da célula com média (mediana/percentil)
   - Threshold: 1.3x maior = mesclada
   - Valida verificando sobreposição com outras células

2. **Marcação:**
   - Célula primária: `is_merged=False`, contém o texto
   - Células secundárias: `is_merged=True`, `merged_from` aponta para primária

3. **Montagem ASCII:**
   - Células secundárias não contribuem com texto (apenas espaço)
   - Células mescladas somam larguras das colunas cobertas
   - Espaçamento entre colunas: 3 caracteres por coluna adicional

