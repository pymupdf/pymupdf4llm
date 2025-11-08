// Copyright (C) 2024 Artifex Software, Inc.
//
// This software may only be used under license.

#include "mupdf/fitz.h"

#include "csv.h"
#include "utils.h"
#include "features_decls.h"

#include <stdio.h>
static int
usage(void)
{
	fprintf(stderr,
		"usage: testfeatures [options] <inputfile> <page range>\n"
		"\t-d -\tThe directory to load PDF files from\n"
		"\n"
		);
	return 1;
}

static void
print_text_in_region(fz_context *ctx, fz_stext_block *block, fz_rect region)
{
	for (; block != NULL; block = block->next)
	{
		fz_stext_line *line;

		if (block->type == FZ_STEXT_BLOCK_STRUCT)
		{
			if (block->u.s.down)
				print_text_in_region(ctx, block->u.s.down->first_block, region);
			continue;
		}
		if (block->type != FZ_STEXT_BLOCK_TEXT)
			continue;
		if (!fz_is_valid_rect(fz_intersect_rect(region, block->bbox)))
			continue;
		for (line = block->u.t.first_line; line != NULL; line = line->next)
		{
			fz_stext_char *ch;
			for (ch = line->first_char; ch != NULL; ch = ch->next)
			{
				fz_rect r = fz_rect_from_quad(ch->quad);
				if (!fz_is_valid_rect(fz_intersect_rect(region, r)))
					continue;
				if (ch->c >= 32 && ch->c <= 255 && ch->c != 127 && ch->c != ',')
					printf("%c", ch->c);
				else
					printf(".");
			}
		}
	}
}


static void
process_region(fz_context *ctx, fz_stext_page *page, fz_features *features, fz_rect region)
{
	fz_feature_stats *stats;

	printf("%g,%g,%g,%g,", region.x0, region.y0, region.x1, region.y1);

	print_text_in_region(ctx, page->first_block, region);

	stats = fz_features_for_region(ctx, features, region, 0);

	/* Output the result */
	printf(",%d,%d,%g,%g,%g,%g,%d,%g,%g,%g,%g,%g,%g,%g,%g,%d,%d,%d,%g,%g,%d,%g,%g,%g,%g,%g,%g,%d,%d,%g,%d,%g,%g,%g,%d,%g,%g,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%g,%g,%g,%g\n",
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
		stats->non_line_bullets,
		stats->consecutive_right_alignment_count_up,
		stats->consecutive_centre_alignment_count_up,
		stats->consecutive_left_alignment_count_up,
		stats->consecutive_right_alignment_count_down,
		stats->consecutive_centre_alignment_count_down,
		stats->consecutive_left_alignment_count_down,
		stats->consecutive_top_alignment_count_left,
		stats->consecutive_middle_alignment_count_left,
		stats->consecutive_bottom_alignment_count_left,
		stats->consecutive_top_alignment_count_right,
		stats->consecutive_middle_alignment_count_right,
		stats->consecutive_bottom_alignment_count_right,
		stats->alignment_up_with_left,
		stats->alignment_up_with_centre,
		stats->alignment_up_with_right,
		stats->alignment_down_with_left,
		stats->alignment_down_with_centre,
		stats->alignment_down_with_right,
		stats->alignment_left_with_top,
		stats->alignment_left_with_middle,
		stats->alignment_left_with_bottom,
		stats->alignment_right_with_top,
		stats->alignment_right_with_middle,
		stats->alignment_right_with_bottom,
		stats->ray_line_distance_up,
		stats->ray_line_distance_down,
		stats->ray_line_distance_left,
		stats->ray_line_distance_right);
}

static void
do_walk_lines(fz_context *ctx, fz_stext_page *page, fz_features *features, fz_stext_block *block)
{
	while (block)
	{
		switch (block->type)
		{
		case FZ_STEXT_BLOCK_STRUCT:
			if (block->u.s.down != NULL)
				do_walk_lines(ctx, page, features, block->u.s.down->first_block);
			break;
		case FZ_STEXT_BLOCK_TEXT:
		{
			fz_stext_line *line;
			for (line = block->u.t.first_line; line != NULL; line = line->next)
			{
				process_region(ctx, page, features, line->bbox);
			}
		}
		}
		
		block = block->next;
	}
}

static void
walk_lines_for_features(fz_context *ctx, fz_stext_page *page, fz_features *features)
{
	do_walk_lines(ctx, page, features, page->first_block);
}

static void
process_page(fz_context *ctx, fz_document *doc, fz_stext_options *options, int page_num)
{
	fz_stext_page *stext = fz_new_stext_page_from_page_number(ctx, doc, page_num-1, options);
	fz_features *features = NULL;

	fz_var(features);

	fz_try(ctx)
	{
		features = fz_new_page_features(ctx, stext);

		walk_lines_for_features(ctx, stext, features);
	}
	fz_always(ctx)
	{
		fz_drop_stext_page(ctx, stext);
		fz_drop_page_features(ctx, features);
	}
	fz_catch(ctx)
		fz_rethrow(ctx);
}

static void
process_range(fz_context *ctx, fz_document *doc, fz_stext_options *options, const char *range)
{
	int page, spage, epage, pagecount;

	pagecount = fz_count_pages(ctx, doc);

	while ((range = fz_parse_page_range(ctx, range, &spage, &epage, pagecount)))
	{
		if (spage < epage)
			for (page = spage; page <= epage; page++)
				process_page(ctx, doc, options, page);
		else
			for (page = spage; page >= epage; page--)
				process_page(ctx, doc, options, page);
	}
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
	char *doc_name = NULL;
	fz_document *doc = NULL;
	fz_stext_options options = { FZ_STEXT_ACCURATE_BBOXES | FZ_STEXT_SEGMENT | FZ_STEXT_COLLECT_VECTORS };
	const char *directory = NULL;
	char *filename;

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

	filename = argv[fz_optind++];

	printf("region.x0");
	printf(",region.y0");
	printf(",region.x1");
	printf(",region.y1");
	printf(",text");
	printf(",num non-numeral chars in area");
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
	printf(",consecutive_right_alignment_count_up");
	printf(",consecutive_centre_alignment_count_up");
	printf(",consecutive_left_alignment_count_up");
	printf(",consecutive_right_alignment_count_down");
	printf(",consecutive_centre_alignment_count_down");
	printf(",consecutive_left_alignment_count_down");
	printf(",consecutive_top_alignment_count_left");
	printf(",consecutive_middle_alignment_count_left");
	printf(",consecutive_bottom_alignment_count_left");
	printf(",consecutive_top_alignment_count_right");
	printf(",consecutive_middle_alignment_count_right");
	printf(",consecutive_bottom_alignment_count_right");
	printf(",alignment_up_with_left");
	printf(",alignment_up_with_centre");
	printf(",alignment_up_with_right");
	printf(",alignment_down_with_left");
	printf(",alignment_down_with_centre");
	printf(",alignment_down_with_right");
	printf(",alignment_left_with_top");
	printf(",alignment_left_with_middle");
	printf(",alignment_left_with_bottom");
	printf(",alignment_right_with_top");
	printf(",alignment_right_with_middle");
	printf(",alignment_right_with_bottom");
	printf(",ray_line_distance_up");
	printf(",ray_line_distance_down");
	printf(",ray_line_distance_left");
	printf(",ray_line_distance_right");
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

	fz_try(ctx)
	{
		doc_name = make_prefixed_name(ctx, directory, filename);
		doc = fz_open_document(ctx, doc_name);

		if (fz_optind == argc)
			process_range(ctx, doc, &options, "1-N");
		else
			process_range(ctx, doc, &options, argv[fz_optind++]);
	}
	fz_always(ctx)
	{
		fz_drop_document(ctx, doc);
		fz_free(ctx, doc_name);
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
