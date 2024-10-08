// Copyright (C) 2024 Artifex Software, Inc.
//
// This software may only be used under license.

#include "mupdf/fitz.h"

#include "csv.h"
#include "utils.h"

#include <stdio.h>

/* This structure, font_freq_list, does double duty, both for
 * fonts, and for linespacings (with fz_font set to NULL). */
typedef struct
{
	fz_font *font;
	float size;
	int freq;
} font_freq_t;

typedef struct
{
	int max;
	int len;
	font_freq_t *list;
} font_freq_list;

static int feq(float a, float b)
{
	a -= b;
	if (a < 0)
		a = -a;
	return (a < 0.01);
}

static void
font_freq_push(fz_context *ctx, font_freq_list *list, fz_font *font, float size)
{
	int i, n = list->len;

	for (i = 0; i < n; i++)
	{
		float fsize = list->list[i].size;
		if (list->list[i].font == font && fsize == size)
		{
			list->list[i].freq++;
			return;
		}
		if (fsize > size)
			break;
	}

	if (list->len == list->max)
	{
		int newmax = list->max * 2;
		if (newmax == 0)
			newmax = 32;
		list->list = fz_realloc(ctx, list->list, newmax * sizeof(list->list[0]));
		list->max = newmax;
	}

	if (i < n)
		memmove(&list->list[i+1], &list->list[i], sizeof(list->list[0]) * (n-i));
	list->list[i].font = fz_keep_font(ctx, font);
	list->list[i].size = size;
	list->list[i].freq = 1;
}

static void
font_freq_common(fz_context *ctx, font_freq_list *list, float delta)
{
	int i, j, n;

	if (list == NULL)
		return;

	/* Common up entries whose size is within delta of each other. */
	n = list->len;
	for (i = 0; i < n - 1; i++)
	{
		float size = list->list[i].size;
		fz_font *f = list->list[i].font;
		for (j = i+1; j < n; j++)
		{
			if (list->list[j].size >= size + delta)
				break;

			if (list->list[j].font == f)
			{
				list->list[i].freq += list->list[j].freq;
				list->list[j].freq = 0;
				/* FIXME: Adjust size */
			}
		}
	}

	/* Remove empty entries */
	for (i = 0, j = 0; i < n; i++)
	{
		if (list->list[i].freq == 0)
			continue;
		if (i != j)
			list->list[j++] = list->list[i];
	}
	list->len = j;

	/* FIXME: If we adjust size above, we'll need to re sort the list. */
}

static int
font_freq_mode(fz_context *ctx, font_freq_list *list)
{
	int i, n, mode, modefreq;

	if (list == NULL || list->len == 0)
		return -1;

	mode = 0;
	modefreq = list->list[0].freq;
	n = list->len;
	for (i = 1; i < n; i++)
	{
		if (list->list[i].freq > modefreq)
			modefreq = list->list[i].freq, mode = i;
	}

	return mode;
}

static int
font_freq_closest(fz_context *ctx, font_freq_list *list, fz_font *font, float size)
{
	int i, n, best;
	float delta = 99999999.0f;

	if (list == NULL || list->len == 0)
		return -1;

	n = list->len;
	best = -1;
	for (i = 0; i < n; i++)
	{
		float f;
		if (list->list[i].font != font)
			continue;
		f = list->list[i].size;
		if (f < size)
		{
			delta = size - f;
			best = i;
		}
		else if (f >= size)
		{
			if (f - size < delta)
			{
				delta = f - size;
				best = i;
			}
			break;
		}
	}

	return best;
}

static void
font_freq_drop(fz_context *ctx, font_freq_list *list)
{
	int i, n;

	if (list == NULL)
		return;

	n = list->len;
	for (i = 0; i < n; i++)
		fz_drop_font(ctx, list->list[i].font);

	fz_free(ctx, list->list);
	list->max = list->len = 0;
}

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
	float line_space;
	int line_space_n = 0;
	font_freq_list fonts = { 0 };
	font_freq_list region_fonts = { 0 };
	font_freq_list linespaces = { 0 };
	font_freq_list region_linespaces = { 0 };
	fz_rect last_char_rect;
	int first_char;
	float last_baseline;
	int first_line;
	float font_size = 0;
	int font_size_n = 0;
	int fonts_mode, linespaces_mode;
	int rfonts_mode, rlinespaces_mode;
	int fonts_offset, linespaces_offset;
	int num_underlines = 0;
	float ratio;
	float margin_l = 50;
	float margin_r = 50;
	float margin_t = 50;
	float margin_b = 50;
	float imargin_l = 50;
	float imargin_r = 50;
	float imargin_t = 50;
	float imargin_b = 50;
	float top_left_x = x1, top_left_y = y1;
	float top_left_height = 0;
	float bottom_right_x = x0, bottom_right_y = y0;
	float bottom_right_height = 0;
	int num_lines = 0;
	float max_non_first_left_indent = 0;
	float max_non_last_right_indent = 0;
	float max_right_indent = 0;

	fz_stext_block *block;
	fz_stext_line *line;
	fz_stext_char *ch;

	fz_rect region = { x0, y0, x1, y1 };

	/* Collect global stats */
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
				line_space = baseline - last_baseline;
				font_freq_push(ctx, &linespaces, NULL, line_space);
			}
			first_line = 0;
			last_baseline = baseline;
			for (ch = line->first_char; ch != NULL; ch = ch->next)
			{
				font_freq_push(ctx, &fonts, ch->font, ch->size);
			}
		}
	}

	/* Now collect for the region itself. */
	line_space = 0;
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
				font_freq_push(ctx, &region_linespaces, NULL, line_space);
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
					float m;

					/* To calculate the indents at top left/bottom right, we need to
					 * look at line y0/y1, not char y0/y1, because otherwise alternating
					 * low and high chars may give confusing results. For example "_'".
					 * The "'" might appear to be in a higher line than the "_". */

					/* If we find a line higher than we've found before (by at least
					 * half the lineheight, to avoid silly effects), recalculate our
					 * top_left_corner. */
					if (line->bbox.y0 < top_left_y - top_left_height/2)
					{
						top_left_y = line->bbox.y0;
						top_left_height = line->bbox.y1 - line->bbox.y0;
						top_left_x = char_rect.x0;
					}
					/* Otherwise if we're in the same line, and x is less, take that. */
					else if (line->bbox.y0 < top_left_y + top_left_height/2 && char_rect.x0 < top_left_x)
					{
						top_left_x = char_rect.x0;
					}

					/* If we find a line lower than we've found before (by at least
					 * half the lineheight, to avoid silly effects)... */
					if (line->bbox.y1 > bottom_right_y + bottom_right_height/2)
					{
						/* recalculate our bottom right corner */
						bottom_right_y = line->bbox.y1;
						bottom_right_height = line->bbox.y1 - line->bbox.y0;
						bottom_right_x = char_rect.x1;
						/* Increment the number of lines, and prepare the indents. */
						num_lines++;
						if (num_lines > 1 && max_non_first_left_indent < line->bbox.x0 - region.x0)
							max_non_first_left_indent = line->bbox.x0 - region.x0;
						if (max_right_indent > max_non_last_right_indent)
							max_non_last_right_indent = max_right_indent;
						max_right_indent = region.x1 - line->bbox.x1;
					}
					/* Otherwise if we're in the same line... */
					else if (line->bbox.y1 < bottom_right_y + bottom_right_height/2)
					{
						/* if x is greater, take that. */
						if (char_rect.x1 > bottom_right_x)
							bottom_right_x = char_rect.x1;
						/* shrink the right ident we've found so far on this line. */
						max_right_indent = region.x1 - line->bbox.x1;
					}

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

					if (ch->flags & FZ_STEXT_UNDERLINE)
						num_underlines++;

					font_freq_push(ctx, &fonts, ch->font, ch->size);

					font_size += ch->size;
					font_size_n++;

					/* Allow for chars that overlap the edge */
					/* Bit of a hack this. The coords are fed into this program accurate to 3
					 * decimal places. This can round them down. This upsets our margin
					 * calculations, as chars can be seen to extend fractionally outside
					 * the region. Accordingly, we extend the region slightly to avoid this. */
#define ROUND 0.001f
					if (char_rect.x0 < x0 - ROUND)
						margin_l = 0;
					if (char_rect.x1 > x1 + ROUND)
						margin_r = 0;
					if (char_rect.y0 < y0 - ROUND)
						margin_t = 0;
					if (char_rect.y1 > y1 + ROUND)
						margin_b = 0;

					/* Inner margins */
					m = char_rect.x0 - x0;
					if (m < 0)
						m = 0;
					if (imargin_l > m)
						imargin_l = m;
					m = x1 - char_rect.x1;
					if (m < 0)
						m = 0;
					if (imargin_r > m)
						imargin_r = m;
					m = char_rect.y0 - y0;
					if (m < 0)
						m = 0;
					if (imargin_t > m)
						imargin_t = m;
					m = y1 - char_rect.y1;
					if (m < 0)
						m = 0;
					if (imargin_b > m)
						imargin_b = m;
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

	font_freq_common(ctx, &fonts, 0.5);
	font_freq_common(ctx, &linespaces, 0.5);
	font_freq_common(ctx, &region_fonts, 0.5);
	font_freq_common(ctx, &region_linespaces, 0.5);

	fonts_offset = 0;
	fonts_mode = font_freq_mode(ctx, &fonts);
	if (fonts_mode != -1)
	{
		rfonts_mode = font_freq_mode(ctx, &region_fonts);
		if (rfonts_mode != -1)
			fonts_offset = font_freq_closest(ctx, &fonts, region_fonts.list[rfonts_mode].font, region_fonts.list[rfonts_mode].size) - fonts_mode;
	}
	linespaces_offset = 0;
	linespaces_mode = font_freq_mode(ctx, &linespaces);
	if (linespaces_mode != -1)
	{
		rlinespaces_mode = font_freq_mode(ctx, &region_linespaces);
		if (rlinespaces_mode != -1)
			linespaces_offset = font_freq_closest(ctx, &fonts, NULL, region_fonts.list[rlinespaces_mode].size) - fonts_mode;
	}

	/* Horrible, but will turn region_fonts into a list of the unique fonts in this region. */
	font_freq_common(ctx, &region_fonts, 100000);

	top_left_x -= x0;
	if (top_left_x < 0)
	{
		/* We should only ever be negative by a rounding error. */
		assert(top_left_x > -0.01);
		top_left_x = 0;
	}
	bottom_right_x = x1 - bottom_right_x;
	if (bottom_right_x < 0)
	{
		/* We should only ever be negative by a rounding error. */
		assert(bottom_right_x > -0.01);
		bottom_right_x = 0;
	}

	/* Output the result */
	printf("%d,%d,%g,%g,%g,%g,%d,%g,%g,%g,%g,%g,%g,%g,%g,%d,%d,%d,%g,%g,%d,%g,%g\n",
		num_non_numerals, num_numerals, ratio, char_space, line_space, font_size, region_fonts.len,
		margin_l, margin_r, margin_t, margin_b,
		imargin_l, imargin_r, imargin_t, imargin_b,
		fonts_offset, linespaces_offset,
		num_underlines,
		top_left_x, bottom_right_x,
		num_lines, max_non_first_left_indent, max_non_last_right_indent);

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

	font_freq_drop(ctx, &fonts);
	font_freq_drop(ctx, &region_fonts);
	font_freq_drop(ctx, &linespaces);
	font_freq_drop(ctx, &region_linespaces);
}

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
	fz_document *doc = NULL;
	fz_stext_page *stext = NULL;
	char *csvfile;
	FILE *infile;
	fz_stext_options options = { FZ_STEXT_ACCURATE_BBOXES };
	char **csv;
	int page_num, last_page_num = -1;
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
	printf("left inner margin,");
	printf("right inner margin,");
	printf("top inner margin,");
	printf("bottom inner margin,");
	printf("font offset from most popular font,");
	printf("linespace offset from most popular linespace,");
	printf("number of underlined chars,");
	printf("top left indent,");
	printf("bottom right indent,");
	printf("num lines,");
	printf("max non first left indent,");
	printf("max non last right indent\n");

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
				if (doc_name != NULL && strcmp(doc_name, line[0]))
				{
					fz_drop_document(ctx, doc);
					doc = NULL;
					fz_free(ctx, doc_name);
					doc_name = NULL;
				}
				if (doc_name == NULL)
				{
					doc_name = make_prefixed_name(ctx, directory, line[0]);
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
				}

				for (i = 0; i < n; i++)
					printf("%s,", line[i]);
				extract_features(ctx, stext, atof(line[2]), atof(line[3]), atof(line[4]), atof(line[5]), atoi(line[6]));
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
