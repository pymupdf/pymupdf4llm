// Copyright (C) 2024 Artifex Software, Inc.
//
// This software may only be used under license.

#include "mupdf/fitz.h"

#include "csv.h"
#include "utils.h"

#include <stdio.h>

typedef enum
{
	UNKNOWN = 0,
	KEPT = 1,
	DROPPED = 2
} region_state;

typedef struct
{
	fz_rect bbox;
	int type;
	float score;
	int order;
	int state;
} region;

typedef struct
{
	int max;
	int len;
	region *list;
} region_list;

static void
region_list_push(fz_context *ctx, region_list *list, fz_rect rect, int type, float score, int order)
{
	if (list->len == list->max)
	{
		int newmax = list->max * 2;
		if (newmax == 0)
			newmax = 32;
		list->list = fz_realloc(ctx, list->list, sizeof(list->list[0]) * newmax);
		list->max = newmax;
	}
	list->list[list->len].bbox = rect;
	list->list[list->len].type = type;
	list->list[list->len].score = score;
	list->list[list->len].order = order;
	list->list[list->len++].state = UNKNOWN;
}

static int
usage(void)
{
	fprintf(stderr,
		"usage: picker [options] csvfile\n"
		"\t-d -\tThe directory to load PDF files from\n"
		"\n"
		"The CSV file is read line by line; regions for the same page\n"
		"in the same file are collected together, and a selection is\n"
		"made based upon the scores. The output is then created from this.\n"
		"\n"
		"The CSV file is expected to be of the order:\n"
		" - PDF filename\n"
		" - Page number\n"
		" - minimum x coordinate for region\n"
		" - minimum u coordinate for region\n"
		" - maximum x coordinate for region\n"
		" - maximum y coordinate for region\n"
		" - class\n"
		" - score\n"
		" - order\n"
		);
	return 1;
}

static int
cmpregion(void const *a_, void const *b_)
{
	const region *a = a_;
	const region *b = b_;

	return (a->order - b->order);
}

static void
output_content_for_region(fz_context *ctx, fz_stext_page *stext, fz_rect bbox, int type)
{
	int header = 0;
	fz_stext_block *block;
	fz_stext_line *line;
	fz_stext_char *ch;

	for (block = stext->first_block; block != NULL; block = block->next)
	{
		switch (block->type)
		{
		case FZ_STEXT_BLOCK_IMAGE:
			printf("<IMAGE/>");
			break;
		case FZ_STEXT_BLOCK_TEXT:
			for (line = block->u.t.first_line; line != NULL; line = line->next)
			{
				int printed = 0;
				for (ch = line->first_char; ch != NULL; ch = ch->next)
				{
					fz_point p;
					p.x = (ch->quad.ll.x + ch->quad.lr.x + ch->quad.ur.x + ch->quad.ul.x)/4;
					p.y = (ch->quad.ll.y + ch->quad.lr.y + ch->quad.ur.y + ch->quad.ul.y)/4;
					if (fz_is_point_inside_rect(p, bbox))
					{
						if (!header)
						{
							header = 1;
							printf("<TEXT type=%d>\n", type);
						}
						if (ch->c == '<')
							printf("&lt;");
						else if (ch->c == '>')
							printf("&gt;");
						else if (ch->c >= 32 && ch->c < 127)
							printf("%c", ch->c);
						else
							printf("&#x%04x;", ch->c);
						printed = 1;
					}
				}
				if (printed)
					printf("\n");
			}
		}
	}
	if (header)
		printf("\n</TEXT>\n");
}

static void
remove_region_from_region(fz_context *ctx, region_list *regions, fz_rect remove, int i)
{
	fz_rect region = regions->list[i].bbox;
	fz_rect r;
	int type = regions->list[i].type;
	float score = regions->list[i].score;
	int order = regions->list[i].order;

	/* If some of region sticks out above remove, keep it. */
	if (region.y0 < remove.y0 && remove.y0 < region.y1)
	{
		r = region;
		r.y1 = remove.y0;
		if (i >= 0)
		{
			regions->list[i].bbox = r;
			regions->list[i].state = UNKNOWN;
		}
		else
			region_list_push(ctx, regions, r, type, score, order);
		i = -1;
	}
	/* If some of region sticks out to the left of remove, keep it. */
	if (region.x0 < remove.x0 && remove.x0 < region.x1)
	{
		r = region;
		r.x1 = remove.x0;
		if (i >= 0)
		{
			regions->list[i].bbox = r;
			regions->list[i].state = UNKNOWN;
		}
		else
			region_list_push(ctx, regions, r, type, score, order);
		i = -1;
	}
	/* If some of region sticks out to the right of remove, keep it. */
	if (region.x0 < remove.x1 && remove.x1 < region.x1)
	{
		r = region;
		r.x0 = remove.x1;
		if (i >= 0)
		{
			regions->list[i].bbox = r;
			regions->list[i].state = UNKNOWN;
		}
		else
			region_list_push(ctx, regions, r, type, score, order);
		i = -1;
	}
	/* If some of region sticks out below remove, keep it. */
	if (region.y0 < remove.y1 && remove.y1 < region.y1)
	{
		r = region;
		r.y0 = remove.y1;
		if (i >= 0)
		{
			regions->list[i].bbox = r;
			regions->list[i].state = UNKNOWN;
		}
		else
			region_list_push(ctx, regions, r, type, score, order);
		i = -1;
	}
}

static void
resolve_layout(fz_context *ctx, region_list *regions, fz_stext_page *stext)
{
	int i, j, best;
	float best_score;
	int n = regions->len;

	/* First, we pick the highest scoring regions of the page, and discard any
	 * others. */
	while (1)
	{
		/* Find the highest scoring 'UNKNOWN' region, and keep it. */
		best_score = -99999;
		best = -1;
		for (i = 0; i < n; i++)
		{
			if (regions->list[i].state != UNKNOWN)
				continue;
			if (best_score < regions->list[i].score ||
				(best_score == regions->list[i].score && regions->list[best].order > regions->list[i].order))
			{
				best = i;
				best_score = regions->list[i].score;
			}
		}
		/* There wasn't one? We're done. */
		if (best == -1)
			break;
		regions->list[best].state = KEPT;

		/* Now, remove all the UNKNOWN regions that intersected with that one. */
		for (i = 0; i < n; i++)
		{
			if (regions->list[i].state != UNKNOWN)
				continue;
			if (!fz_is_empty_rect(fz_intersect_rect(regions->list[best].bbox, regions->list[i].bbox)))
			{
				regions->list[i].state = DROPPED;
				printf("Region %d collides with %d\n", best, i);
				remove_region_from_region(ctx, regions, regions->list[best].bbox, i);
				n = regions->len;
			}
		}
	}

	/* Properly drop all the regions. */
	j = 0;
	for (i = 0; i < n; i++)
	{
		if (regions->list[i].state == KEPT)
		{
			if (i != j)
				regions->list[j] = regions->list[i];
			j++;
		}
	}
	n = j;
	regions->len = n;

	/* Now, we sort the regions into order. */
	qsort(regions->list, n, sizeof(regions->list[0]), cmpregion);

	/* Now output the content */
	for (i = 0; i < n; i++)
		output_content_for_region(ctx, stext, regions->list[i].bbox, regions->list[i].type);

	/* FIXME: Output any leftovers? */
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
	int n, n_headers;
	char *doc_name = NULL;
	fz_document *doc = NULL;
	fz_stext_page *stext = NULL;
	char *csvfile;
	FILE *infile;
	fz_stext_options options = { 0 };
	char **csv;
	int page_num, last_page_num = -1;
	region_list regions = { 0 };
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
					if (stext)
					{
						resolve_layout(ctx, &regions, stext);
						fz_drop_stext_page(ctx, stext);
						stext = NULL;
						last_page_num = -1;
					}

					stext = fz_new_stext_page_from_page_number(ctx, doc, page_num, &options);
					last_page_num = page_num;
				}

				{
					fz_rect bbox =  { atof(line[2]), atof(line[3]), atof(line[4]), atof(line[5]) };
					region_list_push(ctx, &regions, bbox, atoi(line[6]), atof(line[7]), atoi(line[8]));
				}
			}
			fz_catch(ctx)
				// ignore error and continue
				fz_report_error(ctx);
		}
		if (stext)
			resolve_layout(ctx, &regions, stext);
	}
	fz_always(ctx)
	{
		fz_drop_document(ctx, doc);
		fz_free(ctx, doc_name);
		fz_drop_stext_page(ctx, stext);
		fz_free(ctx, regions.list);
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
