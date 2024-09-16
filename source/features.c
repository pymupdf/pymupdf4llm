// Copyright (C) 2024 Artifex Software, Inc.
//
// This software may only be used under license.

#include "mupdf/fitz.h"
#include "csv.h"

#include <stdio.h>

typedef struct
{
	int max;
	int len;
	fz_font **list;
} font_list;

static void
extract_features(fz_context *ctx, fz_stext_page *page, float x0, float y0, float x1, float y1, int category)
{
	// The number of characters in the area
	// The number of numerals in the area
	// Ratio of characters to area.
	// Average space between characters
	// Average space between text lines
	// Average font size
	// Number of fonts used in the area
	int num_non_numerals = 0;
	int num_numerals = 0;
	int num_chars = 0;
	float char_space = 0;
	int char_space_n = 0;
	float line_space = 0;
	int line_space_n = 0;
	font_list fonts = { 0 };
	fz_rect last_char_rect;
	int first_char;
	float last_baseline;
	int first_line;
	float font_size = 0;
	int font_size_n = 0;
	int i;
	float ratio;
	float margin_l = 50;
	float margin_r = 50;
	float margin_t = 50;
	float margin_b = 50;

	fz_stext_block *block;
	fz_stext_line *line;
	fz_stext_char *ch;

	fz_rect region = { x0, y0, x1, y1 };

	first_line = 1;
	for (block = page->first_block; block != NULL; block = block->next)
	{
		if (block->type != FZ_STEXT_BLOCK_TEXT)
			continue;
		first_char = 1;
		for (line = block->u.t.first_line; line != NULL; line = line->next)
		{
			float baseline = line->first_char->origin.y;
			if (!first_line)
			{
				line_space += baseline - last_baseline;
				line_space_n++;
			}
			first_line = 0;
			last_baseline = baseline;
			for (ch = line->first_char; ch != NULL; ch = ch->next)
			{
				/* Use the centre of the char for bbox comparison */
				fz_point p;
				fz_rect char_rect = fz_rect_from_quad(ch->quad);
				fz_rect intersect = fz_intersect_rect(region, char_rect);
				p.x = (char_rect.x0 + char_rect.x1)/2;
				p.y = (char_rect.y0 + char_rect.y1)/2;
				if (fz_is_valid_rect(intersect))
				{
					/* We have a char to consider */
					if ((ch->c >= '0' && ch->c <= '9') || ch->c == '.' || ch->c == '%')
						num_numerals++;
					else
						num_non_numerals++;
					num_chars++;
					if (!first_char)
					{
						char_space_n++;
						if (last_char_rect.x1 < char_rect.x0)
							char_space += char_rect.x0 - last_char_rect.x1;
					}
					first_char = 0;
					last_char_rect = char_rect;

					font_size += ch->size;
					font_size_n++;
					for (i = 0; i < fonts.len; i++)
					{
						if (fonts.list[i] == ch->font)
							break;
					}
					if (i == fonts.len)
					{
						/* New font */
						if (fonts.len == fonts.max)
						{
							int newmax = fonts.max * 2;
							if (newmax == 0)
								newmax = 10;
							fonts.list = fz_realloc(ctx, fonts.list, sizeof(fonts.list[0]) * newmax);
							fonts.max = newmax;
						}
						fonts.list[fonts.len++] = fz_keep_font(ctx, ch->font);
					}
				}
				else
				{
					if (y0 <= p.y && p.y <= y1)
					{
						float m = x0 - char_rect.x1;
						if (m >= 0 && margin_l > m)
							margin_l = m;
						m = char_rect.x0 - x1;
						if (m >= 0 && margin_r > m)
							margin_r = m;
					}
					if (x0 <= p.x && p.x <= x1)
					{
						float m = y0 - char_rect.y1;
						if (m >= 0 && margin_t > m)
							margin_t = m;
						m = char_rect.y0 - y1;
						if (m >= 0 && margin_b > m)
							margin_b = m;
					}
				}
			}
		}
	}
	if (char_space_n)
		char_space /= char_space_n;
	if (line_space_n)
		line_space /= line_space_n;

	ratio = num_chars / ((x1-x0) * (y1-y0));
	if (font_size_n)
		font_size /= font_size_n;

	/* Output the result */
	printf("%d,%d,%g,%g,%g,%g,%d,%g,%g,%g,%g\n",
		num_non_numerals, num_numerals, ratio, char_space, line_space, font_size, fonts.len,
		margin_l, margin_r, margin_t, margin_b);

#if 0
	fprintf(stderr, "1 0 0 setrgbcolor\n");
	fprintf(stderr, "%g %g moveto %g %g lineto %g %g lineto %g %g lineto %g %g lineto stroke\n",
		x0-margin_l, y0, x0, y0, x0, y1, x0-margin_l, y1, x0-margin_l, y0);
	fprintf(stderr, "%g %g moveto %g %g lineto %g %g lineto %g %g lineto %g %g lineto stroke\n",
		x1, y0, x1+margin_r, y0, x1+margin_r, y1, x1, y1, x1, y0);
	fprintf(stderr, "%g %g moveto %g %g lineto %g %g lineto %g %g lineto %g %g lineto stroke\n",
		x0, y0, x1, y0, x1, y0-margin_t, x0, y0-margin_t, x0, y0);
	fprintf(stderr, "%g %g moveto %g %g lineto %g %g lineto %g %g lineto %g %g lineto stroke\n",
		x0, y1, x1, y1, x1, y1+margin_b, x0, y1+margin_b, x0, y1);
	fprintf(stderr, "0 1 0 setrgbcolor\n");
	fprintf(stderr, "%g %g moveto %g %g lineto %g %g lineto %g %g lineto %g %g lineto stroke\n",
		x0, y0, x1, y0, x1, y1, x0, y1, x0, y0);
#endif

	for (i = 0; i < fonts.len; i++)
		fz_drop_font(ctx, fonts.list[i]);
	fz_free(ctx, fonts.list);
}

static int
usage(void)
{
	fprintf(stderr,
		"usage: features [options] csvfile\n"
		"\t-[No options currently]\n"
		"\n"
		"The CSV file should start with a header line, and each line\n"
		"should contain 6 fields:\n"
		" - PDF filename\n"
		" - minimum x coordinate for region\n"
		" - minimum u coordinate for region\n"
		" - maximum x coordinate for region\n"
		" - maximum y coordinate for region\n"
		" - class (ignored, passed through unchanged)\n"
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
	int n;
	char *doc_name = NULL;
	fz_document *doc = NULL;
	fz_stext_page *stext = NULL;
	char *csvfile;
	FILE *infile;
	fz_stext_options options = { 0 };
	char **csv;

	fz_var(doc);

	while ((c = fz_getopt(argc, argv, "")) != -1)
	{
		switch (c)
		{
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

	csv = csv_read_line(infile, &n);
	if (n != 6)
	{
		fprintf(stderr, "Malformed line\n");
		return 1;
	}
	printf("%s,", csv[0]);
	printf("%s,", csv[1]);
	printf("%s,", csv[2]);
	printf("%s,", csv[3]);
	printf("%s,", csv[4]);
	printf("%s,", csv[5]);
	printf("number of non-numeral chars in area,");
	printf("number of numeral chars,");
	printf("ratio of chars to area,");
	printf("average char space,");
	printf("average line space,");
	printf("average font size,");
	printf("number of fonts in area,");
	printf("left margin,");
	printf("right margin,");
	printf("top margin,");
	printf("bottom margin,");

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
		while (!feof(infile))
		{
			line = csv_read_line(infile, &n);
			if (n == 0)
				continue;
			if (n != 6)
			{
				fprintf(stderr, "Malformed line\n");
				continue;
			}

			fz_try(ctx)
			{
				if (doc_name != NULL && strcmp(doc_name, line[0]))
				{
					fz_drop_document(ctx, doc);
					doc = NULL;
					fz_free(ctx, doc_name);
					doc_name = NULL;
				}
				if (doc_name == NULL)
				{
					doc_name = fz_strdup(ctx, line[0]);
					doc = fz_open_document(ctx, doc_name);

					fz_drop_stext_page(ctx, stext);
					stext = NULL;

					stext = fz_new_stext_page_from_page_number(ctx, doc, 0, &options);
				}

				printf("%s,", line[0]);
				printf("%s,", line[1]);
				printf("%s,", line[2]);
				printf("%s,", line[3]);
				printf("%s,", line[4]);
				printf("%s,", line[5]);
				extract_features(ctx, stext, atof(line[1]), atof(line[2]), atof(line[3]), atof(line[4]), atoi(line[5]));
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
