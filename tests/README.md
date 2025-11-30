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

A estrutura atual é:

```
pymupdf4llm/
├── tests/
│   ├── pymupdf4llm/
│   │   ├── tables/
│   │   │   ├── test_tabela1_matrizpt1.py
│   │   │   ├── test_tabela1_matrizpt2.py
│   │   │   ├── test_tabela5_matrizpt1.py
│   │   │   ├── test_tabela5_matrizpt2.py
│   │   │   ├── test_tabela12_matrizpt1.py
│   │   │   ├── test_tabela12_matrizpt2.py
│   │   │   ├── test_tabela14_matrizpt1.py
│   │   │   └── test_tabela14_matrizpt2.py
│   │   └── llama_index/
│   │       └── test_pdf_markdown_reader.py
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

### Executar Todos os Testes de Tabelas

Na raiz do projeto `pymupdf4llm/`, execute:

```bash
pytest tests/pymupdf4llm/tables/
```

Ou, se você estiver dentro da pasta `tests/`:

```bash
pytest pymupdf4llm/tables/
```a

### Executar Todos os Testes (incluindo outros testes)

Para executar todos os testes do projeto:

```bash
pytest tests/
```

### Executar com Saída Detalhada

Para ver mais informações durante a execução:

```bash
pytest tests/pymupdf4llm/tables/ -v
```

### Executar com Prints Visíveis

Para ver as mensagens de print dos testes (útil para debug):

```bash
pytest tests/pymupdf4llm/tables/ -v -s
```

### Executar um Teste Específico

Para executar apenas um arquivo de teste:

```bash
pytest tests/pymupdf4llm/tables/test_tabela1_matrizpt1.py -v
```


##  Solução de Problemas

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



