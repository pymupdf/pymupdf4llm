// Copyright (C) 2024 Artifex Software, Inc.
//
// This software may only be used under license.

#include "mupdf/fitz.h"

#include "csv.h"
#include "utils.h"

#include <stdio.h>

static int
do_output_regions(fz_context *ctx, fz_stext_page *page, const char *filename, int pagenum, fz_stext_block *block, int i)
{
	for (; block != NULL; block = block->next)
	{
		switch (block->type)
		{
		case FZ_STEXT_BLOCK_TEXT:
			printf("%s,%d,", filename, pagenum);
			printf("%g,%g,%g,%g,0,0,%d\n", block->bbox.x0, block->bbox.y0, block->bbox.x1, block->bbox.y1, i++);
			break;
		case FZ_STEXT_BLOCK_STRUCT:
			if (block->u.s.down)
				i = do_output_regions(ctx, page, filename, pagenum, block->u.s.down->first_block, i);
			break;
		}
	}
	return i;
}

static void
output_regions(fz_context *ctx, fz_stext_page *page, const char *filename, int pagenum)
{
	(void)do_output_regions(ctx, page, filename, pagenum, page->first_block, 0);
}

static int
usage(void)
{
	fprintf(stderr,
		"usage: candidates [options] pdffile\n"
		"\t-d -\tThe directory to load PDF files from\n"
		"\n"
		"The pdffile is read, and candidate regions are produced\n"
		"for all the pages as a CSV file to stdout:\n"
		" - PDF filename\n"
		" - Page number\n"
		" - minimum x coordinate for region\n"
		" - minimum u coordinate for region\n"
		" - maximum x coordinate for region\n"
		" - maximum y coordinate for region\n"
		" - class (always 0, currently)\n"
		" - score (always 0, currently)\n"
		" - order\n"
		);
	return 1;
}

#ifdef _MSC_VER
static int main_char
#else
int main
#endif
(int argc, char *argv[])
{
	int i, n, c;
	fz_context *ctx;
	char *doc_name = NULL;
	fz_document *doc = NULL;
	fz_stext_page *stext = NULL;
	char *filename;
	fz_stext_options options = { 0 };
	const char *directory = NULL;

	fz_var(doc);

	while ((c = fz_getopt(argc, argv, "d:")) != -1)
	{
		switch (c)
		{
		case 'd': directory = fz_optarg; break;
		default: return usage();
		}
	}

	if (fz_optind == argc)
		return usage();

	filename = argv[fz_optind];

	ctx = fz_new_context(NULL, NULL, FZ_STORE_DEFAULT);
	if (!ctx)
	{
		fprintf(stderr, "cannot initialise context\n");
		exit(1);
	}

	fz_register_document_handlers(ctx);

	fz_var(doc_name);
	fz_var(doc);
	fz_var(stext);

	fz_try(ctx)
	{
		doc_name = make_prefixed_name(ctx, directory, filename);
		doc = fz_open_document(ctx, doc_name);

		n = fz_count_pages(ctx, doc);

		options.flags = FZ_STEXT_ACCURATE_BBOXES | FZ_STEXT_PARAGRAPH_BREAK | FZ_STEXT_CLIP | FZ_STEXT_SEGMENT | FZ_STEXT_STRIKEOUT;
		for (i = 0; i < n; i++)
		{
			stext = fz_new_stext_page_from_page_number(ctx, doc, 0, &options);

			if (i == 0)
			{
				printf("PDF filename,");
				printf("page number,");
				printf("min x,");
				printf("min y,");
				printf("max x,");
				printf("max y,");
				printf("class,");
				printf("score,");
				printf("order\n");
			}

			output_regions(ctx, stext, filename, i);

			fz_drop_stext_page(ctx, stext);
			stext = NULL;
		}
	}
	fz_always(ctx)
	{
		fz_drop_document(ctx, doc);
		fz_free(ctx, doc_name);
		fz_drop_stext_page(ctx, stext);
	}
	fz_catch(ctx)
	{
		fz_report_error(ctx);
		fprintf(stderr, "Feature extraction failed\n");
	}

	fz_drop_context(ctx);
	
	return 0;
}


#ifdef _MSC_VER
int wmain(int argc, wchar_t *wargv[])
{
	char **argv = fz_argv_from_wargv(argc, wargv);
	int ret = main_char(argc, argv);
	fz_free_argv(argc, argv);
	return ret;
}
#endif
