// Copyright (C) 2024 Artifex Software, Inc.
//
// This software may only be used under license.

#include "mupdf/fitz.h"

#include "csv.h"
#include "utils.h"
#include "features.h"

#include <stdio.h>
static int
usage(void)
{
	fprintf(stderr,
		"usage: features [options] csvfile\n"
		"\t-d -\tThe directory to load PDF files from\n"
		"\n"
		"The CSV file should start with a header line, and each line\n"
		"should contain at least 6 fields:\n"
		" - PDF filename\n"
		" - Page number\n"
		" - minimum x coordinate for region\n"
		" - minimum u coordinate for region\n"
		" - maximum x coordinate for region\n"
		" - maximum y coordinate for region\n"
		" - class (ignored, passed through unchanged)\n"
		" - score (ignored, passed through unchanged)\n"
		" - order (ignored, passed through unchanged)\n"
		" Any other fields are passed through unchanged\n"
		"\n"
		"The updated CSV is written to stdout. Each line from the input\n"
		"is output with feature details appended.\n"
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
	int c;
	fz_context *ctx;
	char **line;
	int i, n, n_headers;
	char *doc_name = NULL;
	char *last_doc_name = NULL;
	fz_document *doc = NULL;
	fz_stext_page *stext = NULL;
	char *csvfile;
	FILE *infile;
	fz_stext_options options = { FZ_STEXT_ACCURATE_BBOXES | FZ_STEXT_SEGMENT | FZ_STEXT_PARAGRAPH_BREAK };
	char **csv;
	int page_num, last_page_num = -1;
	const char *directory = NULL;
	fz_features *features = NULL;

	fz_var(doc);
	fz_var(features);

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

	csvfile = argv[fz_optind];

	infile = fopen(csvfile, "r");
	if (infile == NULL)
	{
		fprintf(stderr, "Can't read %s\n", csvfile);
		return 1;
	}

	csv = csv_read_line(infile, &n_headers);
	if (n_headers < 6)
	{
		fprintf(stderr, "Malformed line\n");
		return 1;
	}
	for (i = 0; i < n_headers; i++)
		printf("%s,", csv[i]);
	printf("num non-numeral chars in area");
	printf(",num numeral chars");
	printf(",ratio of chars to area");
	printf(",avg char space");
	printf(",avg line space");
	printf(",avg font size");
	printf(",num fonts in area");
	printf(",left margin");
	printf(",right margin");
	printf(",top margin");
	printf(",bottom margin");
	printf(",left inner margin");
	printf(",right inner margin");
	printf(",top inner margin");
	printf(",bottom inner margin");
	printf(",font offset from most popular font");
	printf(",linespace offset from most popular linespace");
	printf(",num underlined chars");
	printf(",top left indent");
	printf(",bottom right indent");
	printf(",num lines");
	printf(",max non first left indent");
	printf(",max non last right indent");
	printf(",left scaled margin");
	printf(",right scaled margin");
	printf(",top scaled margin");
	printf(",bottom scaled margin");
	printf(",dodgy_paragraph_breaks");
	printf(",is_header");
	printf(",context_above_font_size");
	printf(",context_above_is_header");
	printf(",context_above_indent");
	printf(",context_above_outdent");
	printf(",context_below_font_size");
	printf(",context_below_is_header");
	printf(",context_below_indent");
	printf(",context_below_outdent");
	printf(",context_below_bullet");
	printf(",context_header_differs");
	printf(",line_bullets");
	printf(",non_line_bullets");
	printf("\n");

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
	fz_var(last_page_num);

	fz_try(ctx)
	{
		while (!feof(infile))
		{
			line = csv_read_line(infile, &n);
			if (n == 0)
				continue;
			if (n != n_headers)
			{
				fprintf(stderr, "Malformed line - number of fields (%d) does not match number of headers (%d)\n", n, n_headers);
				continue;
			}

			fz_try(ctx)
			{
				fz_rect region;
				fz_feature_stats *stats;

				if (last_doc_name != NULL && strcmp(last_doc_name, line[0]))
				{
					fz_drop_document(ctx, doc);
					doc = NULL;
					fz_free(ctx, doc_name);
					doc_name = NULL;
				}
				if (doc_name == NULL)
				{
					doc_name = make_prefixed_name(ctx, directory, line[0]);
					last_doc_name = doc_name + strlen(doc_name) - strlen(line[0]);
					doc = fz_open_document(ctx, doc_name);
					last_page_num = -1;
				}
				page_num = atoi(line[1]);
				if (page_num != last_page_num)
				{
					fz_drop_stext_page(ctx, stext);
					stext = NULL;
					last_page_num = -1;

					stext = fz_new_stext_page_from_page_number(ctx, doc, page_num, &options);
					last_page_num = page_num;

					fz_drop_page_features(ctx, features);
					features = NULL;
					features = fz_new_page_features(ctx, stext);
				}

				for (i = 0; i < n; i++)
					printf("%s,", line[i]);

				region.x0 = atof(line[2]);
				region.y0 = atof(line[3]);
				region.x1 = atof(line[4]);
				region.y1 = atof(line[5]);
				stats = fz_features_for_region(ctx, features, region, atoi(line[6]));

				/* Output the result */
				printf("%d,%d,%g,%g,%g,%g,%d,%g,%g,%g,%g,%g,%g,%g,%g,%d,%d,%d,%g,%g,%d,%g,%g,%g,%g,%g,%g,%d,%d,%g,%d,%g,%g,%g,%d,%g,%g,%d,%d,%d,%d\n",
					stats->num_non_numerals,
					stats->num_numerals,
					stats->ratio,
					stats->char_space,
					stats->line_space,
					stats->font_size,
					stats->num_fonts_in_region,
					stats->margin_l, stats->margin_r, stats->margin_t, stats->margin_b,
					stats->imargin_l, stats->imargin_r, stats->imargin_t, stats->imargin_b,
					stats->fonts_offset,
					stats->linespaces_offset,
					stats->num_underlines,
					stats->top_left_x,
					stats->bottom_right_x,
					stats->num_lines,
					stats->max_non_first_left_indent,
					stats->max_non_last_right_indent,
					stats->smargin_l,
					stats->smargin_r,
					stats->smargin_t,
					stats->smargin_b,
					stats->dodgy_paragraph_breaks,
					stats->is_header,
					stats->context_above_font_size,
					stats->context_above_is_header,
					stats->context_above_indent,
					stats->context_above_outdent,
					stats->context_below_font_size,
					stats->context_below_is_header,
					stats->context_below_indent,
					stats->context_below_outdent,
					stats->context_below_bullet,
					stats->context_header_differs,
					stats->line_bullets,
					stats->non_line_bullets);

#if 0
				fprintf(stderr, "1 0 0 setrgbcolor\n");
				fprintf(stderr, "%g %g moveto %g %g lineto %g %g lineto %g %g lineto %g %g lineto stroke\n",
					region.x0-stats->margin_l, region.y0, region.x0, region.y0, region.x0, region.y1, region.x0-stats->margin_l, region.y1, region.x0-stats->margin_l, region.y0);
				fprintf(stderr, "%g %g moveto %g %g lineto %g %g lineto %g %g lineto %g %g lineto stroke\n",
					region.x1, region.y0, region.x1+stats->margin_r, region.y0, region.x1+stats->margin_r, region.y1, region.x1, region.y1, region.x1, region.y0);
				fprintf(stderr, "%g %g moveto %g %g lineto %g %g lineto %g %g lineto %g %g lineto stroke\n",
					region.x0, region.y0, region.x1, region.y0, region.x1, region.y0-stats->margin_t, region.x0, region.y0-stats->margin_t, region.x0, region.y0);
				fprintf(stderr, "%g %g moveto %g %g lineto %g %g lineto %g %g lineto %g %g lineto stroke\n",
					region.x0, region.y1, region.x1, region.y1, region.x1, region.y1+stats->margin_b, region.x0, region.y1+stats->margin_b, region.x0, region.y1);
				fprintf(stderr, "0 1 0 setrgbcolor\n");
				fprintf(stderr, "%g %g moveto %g %g lineto %g %g lineto %g %g lineto %g %g lineto stroke\n",
					region.x0, region.y0, region.x1, region.y0, region.x1, region.y1, region.x0, region.y1, region.x0, region.y0);
#endif

			}
			fz_catch(ctx)
				// ignore error and continue
				fz_report_error(ctx);
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
