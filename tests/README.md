# Guia de Testes - Extração de Tabelas de PDFs

Este guia explica como configurar e executar os testes de extração de tabelas de PDFs usando PyMuPDF4LLM.

## Pré-requisitos

Antes de começar, certifique-se de ter:

1. **Python 3.8+** instalado
2. **Ambiente virtual** ativado (recomendado)
3. **Dependências instaladas**:
   - `pytest`
   - `python-dotenv`
   - `pymupdf` (PyMuPDF)
   - `pymupdf4llm` (fork local)

### Instalação das Dependências

Se ainda não instalou as dependências, execute:

```bash
pip install pytest python-dotenv pymupdf
```

## Estrutura de Diretórios

A estrutura recomendada é:

```
pymupdf4llm/
├── tests/
│   ├── pymupdf4llm/
│   │   ├── test_tabela1_matrizpt1.py
│   │   ├── test_tabela1_matrizpt2.py
│   │   └── ...
│   └── README.md (este arquivo)
├── pymupdf4llm/
│   └── pymupdf4llm/
│       └── ...
└── .env (arquivo de configuração - você precisa criar)
```

##  Configuração

### 1. Criar o Arquivo `.env`

Crie um arquivo chamado `.env` na **raiz do projeto** `pymupdf4llm/` (mesmo nível que a pasta `tests/`).



**Conteúdo do arquivo `.env`:**

O arquivo `.env` deve conter apenas uma linha com o caminho completo para o PDF:

```env
PDF_PATH=/caminho/completo/para/seu/arquivo.pdf
```

**Exemplos práticos:**

**Linux/Mac:**
```env
PDF_PATH=/home/blp/Área de trabalho/NeuralTec/primeira execução/Jubilant.pdf
```

**Linux/Mac (com aspas para caminhos com espaços):**
```env
PDF_PATH="/home/blp/Área de trabalho/NeuralTec/primeira execução/Jubilant.pdf"
```

**Windows:**
```env
PDF_PATH=C:\Users\usuario\Documentos\arquivo.pdf
```

**Windows (com barras invertidas duplas):**
```env
PDF_PATH=C:\\Users\\usuario\\Documentos\\arquivo.pdf
```

** Importante:**
- Use o caminho **absoluto** (completo) do arquivo PDF
- No Linux/Mac, caminhos com espaços devem estar entre aspas ou usar escape
- No Windows, use barras normais ou invertidas duplas: `C:\\Users\\...\\arquivo.pdf`
- Não adicione espaços antes ou depois do sinal de `=`
- O arquivo `.env` deve estar na raiz do projeto `pymupdf4llm/`, não dentro da pasta `tests/`

### 2. Onde Colocar os PDFs

Você pode colocar os PDFs em qualquer lugar do sistema. O importante é que o caminho no arquivo `.env` aponte corretamente para o arquivo.

**Sugestão de organização:**

Crie uma pasta para os PDFs de teste:

```bash
mkdir -p pymupdf4llm/tests/pdfs
```

E então coloque seus PDFs lá. No arquivo `.env`:

```env
PDF_PATH=/home/blp/Área de trabalho/NeuralTec/pymupdf4llm/tests/pdfs/Jubilant.pdf
```

## Executando os Testes

### Executar Todos os Testes

Na raiz do projeto `pymupdf4llm/`, execute:

```bash
pytest tests/
```


## Descrição dos Testes

### `test_tabela1_matrizpt1.py`

Este arquivo contém o teste `test_primeira_tabela_com_llm` que:

1. **Extrai a primeira tabela** do PDF usando PyMuPDF4LLM
2. **Tenta diferentes estratégias** de detecção de tabelas:
   - `lines_strict`: Detecção estrita por linhas
   - `lines`: Detecção por linhas (menos estrita)
   - `text`: Detecção por texto
3. **Verifica se a tabela é uma matriz** (lista de listas)
4. **Compara valores específicos** nas posições esperadas:
   - `(0, 0)`: "STAGE : ARP-3"
   - `(0, 1)`: "" (vazio)
   - `(1, 0)`: "Input batch size"
   - `(1, 1)`: "Output batch size"
   - `(2, 0)`: "55 – 60 Kg of ARP2"
   - `(2, 1)`: "43.18 to 57.6"
5. **Mostra informações detalhadas** sobre a estrutura encontrada

### `test_tabela1_matrizpt2.py`

Este arquivo contém dois testes:

#### `test_primeira_tabela_com_llm`

Similar ao teste do arquivo `pt1`, mas com melhor tratamento de células mescladas.

#### `test_matriz_ascii_comparacao_imagem`

Este teste:

1. **Extrai a primeira tabela** do PDF
2. **Converte para formato ASCII** (representação visual com caracteres)
3. **Compara exatamente** com o formato esperado:

```
------------------------------------------
| STAGE : ARP-3                          |
------------------------------------------
| Input batch size   | Output batch size |
------------------------------------------
| 55 – 60 Kg of ARP2 | 43.18 to 57.6     |
------------------------------------------
```

4. **Falha se houver diferenças** mínimas na formatação

## 🔍 Solução de Problemas

### Erro: "Variável de ambiente PDF_PATH não encontrada"

**Causa:** O arquivo `.env` não existe ou não está no local correto.

**Solução:**
1. Verifique se o arquivo `.env` está na raiz do projeto `pymupdf4llm/`
2. Verifique se o arquivo contém a linha `PDF_PATH=...`
3. Certifique-se de que não há espaços antes ou depois do `=`

### Erro: "PDF de teste não encontrado em ..."

**Causa:** O caminho especificado no `.env` está incorreto ou o arquivo não existe.

**Solução:**
1. Verifique se o caminho no `.env` está correto
2. Use caminho absoluto (completo)
3. Verifique se o arquivo PDF realmente existe nesse local
4. No Linux/Mac, você pode verificar com: `ls -la "/caminho/completo/arquivo.pdf"`

### Erro: "Nenhuma tabela foi detectada no PDF"

**Causa:** O PDF não contém tabelas ou as estratégias de detecção não estão funcionando.

**Solução:**
1. Verifique se o PDF realmente contém tabelas
2. Tente abrir o PDF em um visualizador para confirmar
3. Os testes tentam automaticamente diferentes estratégias, mas algumas tabelas podem não ser detectáveis

### Erro de Importação: "No module named 'pymupdf4llm'"

**Causa:** O módulo pymupdf4llm não está instalado ou o caminho está incorreto.

**Solução:**
1. Verifique se você está no ambiente virtual correto
2. Instale o módulo: `pip install -e pymupdf4llm/`
3. Verifique se o caminho no código está correto (linha 11 dos arquivos de teste)

### Erro: Caminho com espaços não funciona

**Causa:** Caminhos com espaços precisam de tratamento especial.

**Solução:**
No arquivo `.env`, use aspas ou escape:

```env
PDF_PATH="/home/usuario/Meus Documentos/arquivo.pdf"
```

Ou:

```env
PDF_PATH=/home/usuario/Meus\ Documentos/arquivo.pdf
```



