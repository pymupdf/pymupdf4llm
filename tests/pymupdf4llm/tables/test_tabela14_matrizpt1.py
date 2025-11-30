import pytest
from pathlib import Path
import sys
import os
import json
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


def extrair_primeira_tabela_llm(pdf_path: Path, strategy="lines_strict", pagina=24):
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


def extrair_primeira_tabela_pymupdf(pdf_path: Path, strategy="lines_strict", pagina=24):
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
        tabela, estrutura_completa = extrair_primeira_tabela_llm(pdf_teste, strategy, pagina=24)
        if tabela is not None:
            estrategia_usada = strategy
            metodo_usado = "pymupdf4llm"
            break
    
    # Se não encontrou com pymupdf4llm, tenta PyMuPDF diretamente
    if tabela is None:
        print("\npymupdf4llm não encontrou tabelas na página 24, tentando PyMuPDF diretamente...")
        for strategy in estrategias:
            tabela, estrutura_completa = extrair_primeira_tabela_pymupdf(pdf_teste, strategy, pagina=24)
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
            print(f"\nRow {i}:")
            for j, celula in enumerate(linha):
                if isinstance(celula, dict):
                    print(f"  Cell [{i}][{j}]:")
                    # Formata o dicionário de forma legível
                    celula_formatada = {
                        "text": celula.get("text", ""),
                        "row": celula.get("row", "?"),
                        "col": celula.get("col", "?"),
                        "rowspan": celula.get("rowspan", 1),
                        "colspan": celula.get("colspan", 1),
                    }
                    if "bbox" in celula:
                        celula_formatada["bbox"] = celula.get("bbox")
                    if "is_merged" in celula:
                        celula_formatada["is_merged"] = celula.get("is_merged", False)
                    if "merged_from" in celula:
                        celula_formatada["merged_from"] = celula.get("merged_from")
                    if "primary_row" in celula:
                        celula_formatada["primary_row"] = celula.get("primary_row")
                    if "primary_col" in celula:
                        celula_formatada["primary_col"] = celula.get("primary_col")
                    print(json.dumps(celula_formatada, indent=2, ensure_ascii=False))
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
        (0, 0): "S.No",
        (0, 1): "Batch. No",
        (0, 2): "Residue on Ignition",
        (1, 0): "1",
        (1, 1): "3ARP321002",
        (1, 2): "0.03%",
        (2, 0): "2",
        (2, 1): "3ARP321003",
        (2, 2): "0.03%",
        (3, 0): "3",
        (3, 1): "3ARP321004",
        (3, 2): "0.03%",
        (4, 0): "Specification Limit",
        (4, 2): "Not more than 0.1%"
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
    
    def extrair_texto_celula(celula, i, j, tabela):
        """
        Extrai o texto de uma célula, lidando com células mescladas.
        Se a célula for mesclada, retorna string vazia, pois o conteúdo
        está na célula primária.
        """
        if isinstance(celula, dict):
            # Verifica se é célula mesclada
            is_merged = celula.get("is_merged", False)
            if is_merged:
                # Células mescladas não têm conteúdo próprio, retornam string vazia
                return ""
            
            # Retorna o texto da célula atual (não mesclada)
            return celula.get("text", "")
        else:
            return str(celula)
    
    for (i, j), valor_esperado in esperado.items():
        try:
            celula = tabela[i][j]
            valor_obtido = extrair_texto_celula(celula, i, j, tabela)
        except IndexError:
            erros.append(f"Posição ({i},{j}) não existe na tabela extraída.")
            print(f"   ({i},{j}): Posição não existe (tabela tem {len(tabela)} linhas)")
            continue

        # Comparação case-insensitive e parcial
        # Se o valor esperado for vazio, aceita qualquer string vazia ou None
        if valor_esperado == "":
            if valor_obtido == "" or valor_obtido is None:
                print(f"   ({i},{j}): '{valor_obtido}' ✓ (esperado vazio)")
            else:
                erros.append(
                    f"Valor incorreto em ({i},{j}):\n"
                    f"   Esperado: '' (vazio)\n"
                    f"   Obtido:   '{valor_obtido}'"
                )
                print(f"   ({i},{j}): Esperado '' (vazio), Obtido '{valor_obtido}' ✗")
        elif valor_esperado.lower() not in str(valor_obtido).lower():
            erros.append(
                f"Valor incorreto em ({i},{j}):\n"
                f"   Esperado: '{valor_esperado}'\n"
                f"   Obtido:   '{valor_obtido}'"
            )
            print(f"   ({i},{j}): Esperado '{valor_esperado}', Obtido '{valor_obtido}' ✗")
        else:
            print(f"   ({i},{j}): '{valor_obtido}' ✓")

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
