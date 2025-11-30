import pytest
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# adiciona o caminho até o segundo nível do pacote
sys.path.insert(0, "/home/blp/Área de trabalho/NeuralTec/pymupdf4llm/pymupdf4llm")

import pymupdf4llm as llm
import fitz  # PyMuPDF



@pytest.fixture
def pdf_teste(tmp_path):
    """
    Define o caminho do PDF de teste.
    O caminho é lido da variável de ambiente PDF_PATH no arquivo .env.
    """
    pdf_path_str = os.getenv("PDF_PATH")
    assert pdf_path_str, "Variável de ambiente PDF_PATH não encontrada no arquivo .env"
    pdf_path = Path(pdf_path_str)
    assert pdf_path.exists(), f"PDF de teste não encontrado em {pdf_path}"
    return pdf_path


def extrair_primeira_tabela_llm(pdf_path: Path, strategy="lines_strict", pagina=14):
    """
    Extrai a primeira tabela da página especificada usando PyMuPDF4LLM (fork local).
    Retorna uma tupla (dados_tabela, estrutura_completa) onde:
    - dados_tabela: lista de listas (matriz) ou None
    - estrutura_completa: dict com informações completas da tabela encontrada
    """
    chunks = llm.to_markdown(str(pdf_path), page_chunks=True, table_strategy=strategy)
    
    # Debug: mostra todos os chunks e suas tabelas
    print(f"\nBuscando tabelas na página {pagina} com estratégia '{strategy}'...")
    print(f"Total de chunks (páginas): {len(chunks)}")
    
    # Ajusta o índice (páginas começam em 1, mas índices em 0)
    idx_pagina = pagina - 1
    
    if idx_pagina < len(chunks):
        ch = chunks[idx_pagina]
        tabelas = ch.get("tables") or []
        print(f"  Página {pagina} (chunk {idx_pagina + 1}): {len(tabelas)} tabela(s) encontrada(s)")
        
        if tabelas:
            tabela = tabelas[0]
            print(f"  Primeira tabela encontrada na página {pagina}")
            
            # Tenta extrair a matriz de diferentes formas
            dados = None
            if "matriz" in tabela:
                dados = tabela["matriz"]
            elif "data" in tabela:
                dados = tabela["data"]
            elif "markdown" in tabela:
                # Se só tiver markdown, tenta converter
                dados = tabela["markdown"]
            else:
                # Retorna a estrutura completa para debug
                dados = tabela
            
            return dados, tabela
    else:
        print(f"  Página {pagina} não existe no PDF (total de páginas: {len(chunks)})")
    
    return None, None


def extrair_primeira_tabela_pymupdf(pdf_path: Path, strategy="lines_strict", pagina=14):
    """
    Extrai a primeira tabela da página especificada usando PyMuPDF diretamente (fallback).
    Retorna uma tupla (dados_tabela, estrutura_completa).
    """
    doc = fitz.open(str(pdf_path))
    
    print(f"\nTentando PyMuPDF diretamente na página {pagina} com estratégia '{strategy}'...")
    
    # Ajusta o índice (páginas começam em 1, mas índices em 0)
    idx_pagina = pagina - 1
    
    if idx_pagina < len(doc):
        page = doc[idx_pagina]
        tables = page.find_tables(strategy=strategy)
        
        if tables.tables:
            print(f"  Tabela encontrada na página {pagina}")
            primeira_tabela = tables.tables[0]
            
            # Converte para matriz
            try:
                matriz = primeira_tabela.extract()
                estrutura = {
                    "bbox": primeira_tabela.bbox,
                    "rows": primeira_tabela.row_count,
                    "cols": primeira_tabela.col_count,
                    "markdown": primeira_tabela.to_markdown()
                }
                doc.close()
                return matriz, estrutura
            except Exception as e:
                print(f"  Erro ao extrair tabela: {e}")
                doc.close()
                return None, None
    else:
        print(f"  Página {pagina} não existe no PDF (total de páginas: {len(doc)})")
    
    doc.close()
    return None, None


def test_primeira_tabela_com_llm(pdf_teste):
    """
    Testa se a primeira tabela lida pelo fork local do PyMuPDF4LLM
    é uma matriz e contém os valores esperados nas posições.
    Mostra o que foi encontrado e o que era esperado, mesmo que não seja uma matriz.
    """
    
    # Tenta diferentes estratégias com pymupdf4llm
    estrategias = ["lines_strict", "lines", "text"]
    tabela = None
    estrutura_completa = None
    estrategia_usada = None
    metodo_usado = None
    
    for strategy in estrategias:
        tabela, estrutura_completa = extrair_primeira_tabela_llm(pdf_teste, strategy, pagina=14)
        if tabela is not None:
            estrategia_usada = strategy
            metodo_usado = "pymupdf4llm"
            break
    
    # Se não encontrou com pymupdf4llm, tenta PyMuPDF diretamente
    if tabela is None:
        print("\npymupdf4llm não encontrou tabelas na página 14, tentando PyMuPDF diretamente...")
        for strategy in estrategias:
            tabela, estrutura_completa = extrair_primeira_tabela_pymupdf(pdf_teste, strategy, pagina=14)
            if tabela is not None:
                estrategia_usada = strategy
                metodo_usado = "pymupdf_direto"
                break
    
    # Mostra o que foi encontrado
    print("\n" + "="*80)
    print("RESULTADO DA EXTRAÇÃO")
    print("="*80)
    
    if tabela is None:
        print("Nenhuma tabela foi detectada no PDF com nenhuma das estratégias.")
        print("\nEstrutura completa dos chunks (para debug):")
        chunks = llm.to_markdown(str(pdf_teste), page_chunks=True, table_strategy="lines_strict")
        for idx, ch in enumerate(chunks):
            print(f"\n  Chunk {idx + 1}:")
            print(f"    Chaves disponíveis: {list(ch.keys())}")
            if "tables" in ch:
                print(f"    Tabelas: {ch['tables']}")
            if "text" in ch:
                texto_preview = ch["text"][:200] if len(ch["text"]) > 200 else ch["text"]
                print(f"    Texto (preview): {texto_preview}...")
        pytest.fail("Nenhuma tabela foi detectada no PDF.")
    
    print(f"Tabela encontrada usando método: '{metodo_usado}' com estratégia: '{estrategia_usada}'")
    print(f"Tipo do objeto retornado: {type(tabela)}")
    
    # Mostra a estrutura completa encontrada
    if estrutura_completa:
        print(f"\nEstrutura completa da tabela:")
        print(f"   Chaves disponíveis: {list(estrutura_completa.keys())}")
        for key, value in estrutura_completa.items():
            if key not in ["matriz", "data"]:  # Já vamos mostrar esses separadamente
                print(f"   {key}: {str(value)[:100]}...")
    
    # Verifica formato e mostra o que foi encontrado
    print(f"\nCONTEÚDO ENCONTRADO:")
    print("-"*80)
    
    is_matriz = isinstance(tabela, list) and all(isinstance(linha, list) for linha in tabela) if isinstance(tabela, list) else False
    
    # Verifica se é matriz de dicionários (nova estrutura)
    is_matriz_dict = False
    if is_matriz and len(tabela) > 0 and len(tabela[0]) > 0:
        primeira_celula = tabela[0][0]
        is_matriz_dict = isinstance(primeira_celula, dict) and "text" in primeira_celula
    
    if is_matriz_dict:
        print("Formato: Matriz de dicionários (nova estrutura com metadados)")
        for i, linha in enumerate(tabela):
            print(f"   Linha {i}:")
            for j, celula in enumerate(linha):
                if isinstance(celula, dict):
                    print(f"      [{i},{j}]: text='{celula.get('text', '')}', row={celula.get('row', '?')}, col={celula.get('col', '?')}, rowspan={celula.get('rowspan', 1)}, colspan={celula.get('colspan', 1)}")
                else:
                    print(f"      [{i},{j}]: {celula}")
    elif is_matriz:
        print("Formato: Matriz (lista de listas)")
        for i, linha in enumerate(tabela):
            print(f"   Linha {i}: {linha}")
    elif isinstance(tabela, list):
        print("Formato: Lista (mas não é matriz)")
        for i, item in enumerate(tabela):
            print(f"   Item {i}: {item} (tipo: {type(item)})")
    elif isinstance(tabela, str):
        print("Formato: String (markdown ou texto)")
        print(f"   Conteúdo:\n{tabela}")
    else:
        print(f"Formato: {type(tabela)}")
        print(f"   Conteúdo: {tabela}")
    
    # Define o resultado esperado
    print(f"\nVALORES ESPERADOS:")
    print("-"*80)
    esperado = {
        (0, 0): "Name of Solvents",
        (0, 1): "Limit",
        (1, 0): "Acetonitrile",
        (1, 1): "Not more than 200 ppm",
        (2, 0): "Isopropyl alcohol",
        (2, 1): "Not more than 2000 ppm",
        (3, 0): "Cyclohexane",
        (3, 1): "Not more than 1000 ppm"
    }
    
    for (i, j), valor_esperado in esperado.items():
        print(f"   ({i},{j}): '{valor_esperado}'")
    
    # Compara apenas se for matriz
    if not is_matriz:
        print(f"\nA tabela extraída não é uma matriz (lista de listas).")
        print(f"   Tipo encontrado: {type(tabela)}")
        print(f"   Conteúdo encontrado: {tabela}")
        pytest.fail(
            f"A tabela extraída não é uma matriz.\n"
            f"Tipo encontrado: {type(tabela)}\n"
            f"Conteúdo: {tabela}\n"
            f"Estrutura completa: {estrutura_completa}"
        )
    
    # Compara valores
    print(f"\nCOMPARAÇÃO:")
    print("-"*80)
    erros = []
    
    for (i, j), valor_esperado in esperado.items():
        try:
            celula = tabela[i][j]
            # Extrai o texto dependendo da estrutura
            if isinstance(celula, dict):
                valor_obtido = celula.get("text", "")
            else:
                valor_obtido = celula
        except IndexError:
            erros.append(f"Posição ({i},{j}) não existe na tabela extraída.")
            print(f"   ({i},{j}): Posição não existe (tabela tem {len(tabela)} linhas)")
            continue

        # Comparação case-insensitive e parcial
        if valor_esperado.lower() not in str(valor_obtido).lower():
            erros.append(
                f"Valor incorreto em ({i},{j}):\n"
                f"   Esperado: '{valor_esperado}'\n"
                f"   Obtido:   '{valor_obtido}'"
            )
            print(f"   ({i},{j}): Esperado '{valor_esperado}', Obtido '{valor_obtido}'")
        else:
            print(f"   ({i},{j}): '{valor_obtido}'")

    # Mostra resumo final
    print("\n" + "="*80)
    if erros:
        print("RESULTADO FINAL: diferenças encontradas")
        print("="*80)
        for e in erros:
            print(e)
        pytest.fail("Alguns valores não conferem com o esperado.")
    else:
        print("RESULTADO FINAL: todos os valores conferem com o esperado.")
        print("="*80)


def test_matriz_ascii_comparacao_imagem(pdf_teste):
    """
    Testa se a matriz ASCII extraída da primeira tabela corresponde
    exatamente ao formato esperado.
    
    O formato esperado é:
    ----------------------------------------------
    | Name of Solvents  | Limit                  |
    ----------------------------------------------
    | Acetonitrile      | Not more than 200 ppm  |
    ----------------------------------------------
    | Isopropyl alcohol | Not more than 2000 ppm |
    ----------------------------------------------
    | Cyclohexane       | Not more than 1000 ppm |
    ----------------------------------------------
    """
    
    # Define o resultado esperado exato
    matriz_ascii_esperada = """----------------------------------------------
| Name of Solvents  | Limit                  |
----------------------------------------------
| Acetonitrile      | Not more than 200 ppm  |
----------------------------------------------
| Isopropyl alcohol | Not more than 2000 ppm |
----------------------------------------------
| Cyclohexane       | Not more than 1000 ppm |
----------------------------------------------"""
    
    # Tenta diferentes estratégias com pymupdf4llm
    estrategias = ["lines_strict", "lines", "text"]
    estrutura_completa = None
    estrategia_usada = None
    metodo_usado = None
    
    for strategy in estrategias:
        chunks = llm.to_markdown(str(pdf_teste), page_chunks=True, table_strategy=strategy)
        
        # Busca especificamente na página 14 (índice 13)
        idx_pagina = 14 - 1
        if idx_pagina < len(chunks):
            ch = chunks[idx_pagina]
            tabelas = ch.get("tables") or []
            if tabelas:
                estrutura_completa = tabelas[0]
                estrategia_usada = strategy
                metodo_usado = "pymupdf4llm"
                break
        
        if estrutura_completa:
            break
    
    # Se não encontrou com pymupdf4llm, tenta PyMuPDF diretamente
    if estrutura_completa is None:
        print("\npymupdf4llm não encontrou tabelas na página 14, tentando PyMuPDF diretamente...")
        for strategy in estrategias:
            doc = fitz.open(str(pdf_teste))
            try:
                # Busca especificamente na página 14 (índice 13)
                idx_pagina = 14 - 1
                if idx_pagina < len(doc):
                    page = doc[idx_pagina]
                    tables = page.find_tables(strategy=strategy)
                    if tables.tables:
                        primeira_tabela = tables.tables[0]
                    matriz = primeira_tabela.extract()
                    # Converte para o formato esperado
                    # Importa a função matriz_to_ascii do módulo helpers
                    # Usa caminho relativo ao projeto
                    base_path = Path(__file__).parent.parent.parent
                    helpers_path = base_path / "pymupdf4llm" / "pymupdf4llm" / "helpers"
                    if str(helpers_path) not in sys.path:
                        sys.path.insert(0, str(helpers_path))
                    from pymupdf_rag import matriz_to_ascii
                    # Converte matriz simples para formato com dicionários se necessário
                    matriz_formatada = []
                    for row_idx, row in enumerate(matriz):
                        matriz_row = []
                        for col_idx, cell in enumerate(row):
                            if isinstance(cell, dict):
                                matriz_row.append(cell)
                            else:
                                matriz_row.append({
                                    "text": str(cell) if cell is not None else "",
                                    "row": row_idx,
                                    "col": col_idx,
                                    "rowspan": 1,
                                    "colspan": 1
                                })
                        matriz_formatada.append(matriz_row)
                    matriz_ascii = matriz_to_ascii(matriz_formatada)
                    estrutura_completa = {
                        "matriz_ascii": matriz_ascii,
                        "matriz": matriz
                    }
                    estrategia_usada = strategy
                    metodo_usado = "pymupdf_direto"
                    break
            finally:
                doc.close()
            if estrutura_completa:
                break
    
    # Mostra o que foi encontrado
    print("\n" + "="*80)
    print("TESTE DE COMPARAÇÃO EXATA DA MATRIZ ASCII")
    print("="*80)
    
    if estrutura_completa is None:
        print("Nenhuma tabela foi detectada no PDF.")
        pytest.fail("Nenhuma tabela foi detectada no PDF.")
    
    print(f"Tabela encontrada usando método: '{metodo_usado}' com estratégia: '{estrategia_usada}'")
    
    # Obtém a matriz ASCII
    matriz_ascii = estrutura_completa.get("matriz_ascii")
    
    if matriz_ascii is None:
        print("\nA tabela não possui o campo 'matriz_ascii'.")
        print(f"Chaves disponíveis na estrutura: {list(estrutura_completa.keys())}")
        pytest.fail("A tabela extraída não possui o campo 'matriz_ascii'.")
    
    # Normaliza ambas as matrizes para comparação (remove espaços em branco no final das linhas)
    matriz_ascii_normalizada = "\n".join(linha.rstrip() for linha in matriz_ascii.split("\n"))
    matriz_ascii_esperada_normalizada = "\n".join(linha.rstrip() for linha in matriz_ascii_esperada.split("\n"))
    
    print(f"\nMATRIZ ASCII ESPERADA:")
    print("-"*80)
    print(matriz_ascii_esperada_normalizada)
    print("-"*80)
    
    print(f"\nMATRIZ ASCII EXTRAÍDA:")
    print("-"*80)
    print(matriz_ascii_normalizada)
    print("-"*80)
    
    # Comparação exata linha por linha
    linhas_esperadas = matriz_ascii_esperada_normalizada.split("\n")
    linhas_obtidas = matriz_ascii_normalizada.split("\n")
    
    print(f"\nCOMPARAÇÃO LINHA POR LINHA:")
    print("-"*80)
    erros = []
    
    # Verifica se o número de linhas é o mesmo
    if len(linhas_obtidas) != len(linhas_esperadas):
        erros.append(
            f"Número de linhas diferente:\n"
            f"   Esperado: {len(linhas_esperadas)} linhas\n"
            f"   Obtido: {len(linhas_obtidas)} linhas"
        )
        print(f"✗ Número de linhas diferente: esperado {len(linhas_esperadas)}, obtido {len(linhas_obtidas)}")
    else:
        print(f"✓ Número de linhas correto: {len(linhas_esperadas)}")
    
    # Compara cada linha
    max_linhas = max(len(linhas_esperadas), len(linhas_obtidas))
    for i in range(max_linhas):
        if i < len(linhas_esperadas) and i < len(linhas_obtidas):
            esperada = linhas_esperadas[i]
            obtida = linhas_obtidas[i]
            if esperada == obtida:
                print(f"✓ Linha {i+1}: OK")
            else:
                erros.append(
                    f"Linha {i+1} diferente:\n"
                    f"   Esperado: '{esperada}'\n"
                    f"   Obtido:   '{obtida}'"
                )
                print(f"✗ Linha {i+1}: DIFERENTE")
                print(f"    Esperado: '{esperada}'")
                print(f"    Obtido:   '{obtida}'")
        elif i < len(linhas_esperadas):
            erros.append(f"Linha {i+1} faltando na matriz extraída. Esperado: '{linhas_esperadas[i]}'")
            print(f"✗ Linha {i+1}: FALTANDO (esperado: '{linhas_esperadas[i]}')")
        else:
            erros.append(f"Linha {i+1} extra na matriz extraída. Obtido: '{linhas_obtidas[i]}'")
            print(f"✗ Linha {i+1}: EXTRA (obtido: '{linhas_obtidas[i]}')")
    
    # Comparação exata completa
    if matriz_ascii_normalizada == matriz_ascii_esperada_normalizada:
        print("\n" + "="*80)
        print("RESULTADO FINAL: matriz ASCII corresponde EXATAMENTE ao formato esperado")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("RESULTADO FINAL: diferenças encontradas entre a matriz ASCII e o formato esperado")
        print("="*80)
        print(f"\nTotal de erros: {len(erros)}")
        print("\nErros encontrados:")
        for e in erros:
            print(f"  - {e}")
        print(f"\nMatriz ASCII esperada:")
        print("-"*80)
        print(matriz_ascii_esperada_normalizada)
        print("-"*80)
        print(f"\nMatriz ASCII obtida:")
        print("-"*80)
        print(matriz_ascii_normalizada)
        print("-"*80)
        pytest.fail(f"A matriz ASCII extraída não corresponde exatamente ao formato esperado.\nTotal de erros: {len(erros)}")
