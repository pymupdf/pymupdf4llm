// Copyright (C) 2024 Artifex Software, Inc.
//
// This software may only be used under license.

// features.h is a special header with gcc.
#include "features_decls.h"

#include "csv.h"
#include "utils.h"

#include <stdio.h>

/* #define DEBUG_CHARS */

#define MAX_MARGIN 50

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

struct fz_features
{
	fz_feature_stats stats;
	font_freq_list fonts;
	font_freq_list linespaces;
	font_freq_list region_fonts;
	font_freq_list region_linespaces;
	fz_stext_page *page;
	int num_chars;
	int font_size_n;
	int fonts_mode;
	int linespaces_mode;
	int rfonts_mode;
	int rlinespaces_mode;
	float top_left_y;
	float top_left_height;
	float bottom_right_y;
	float bottom_right_height;
	float max_right_indent;
};

#if 0
static int feq(float a, float b)
{
	a -= b;
	if (a < 0)
		a = -a;
	return (a < 0.01);
}
#endif

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
		list->list = (font_freq_t*) fz_realloc(ctx, list->list, newmax * sizeof(list->list[0]));
		list->max = newmax;
	}

	if (i < n)
		memmove(&list->list[i+1], &list->list[i], sizeof(list->list[0]) * (n-i));
	list->list[i].font = fz_keep_font(ctx, font);
	list->list[i].size = size;
	list->list[i].freq = 1;
	list->len++;
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
			list->list[j] = list->list[i];
		j++;
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
gather_global_stats(fz_context *ctx, fz_stext_block *block, fz_features *features)
{
	int first_line = 1;
	float last_baseline;

	for (; block != NULL; block = block->next)
	{
		fz_stext_line *line;

		if (block->type == FZ_STEXT_BLOCK_STRUCT)
		{
			if (block->u.s.down)
				gather_global_stats(ctx, block->u.s.down->first_block, features);
			continue;
		}
		else if (block->type != FZ_STEXT_BLOCK_TEXT)
			continue;
		for (line = block->u.t.first_line; line != NULL; line = line->next)
		{
			fz_stext_char *ch;
			float baseline = line->first_char->origin.y;

			if (!first_line && line->first_char)
			{
				float threshold_line_change = line->first_char->size/4;
				float line_space = baseline - last_baseline;
				if (line_space > threshold_line_change || line_space < -threshold_line_change)
					font_freq_push(ctx, &features->linespaces, NULL, line_space);
			}
			first_line = 0;
			last_baseline = baseline;
			for (ch = line->first_char; ch != NULL; ch = ch->next)
			{
				font_freq_push(ctx, &features->fonts, ch->font, ch->size);
			}
		}
	}
}

static int
struct_is_header(fz_stext_struct *s)
{
	if (s == NULL)
		return 0;
	
	return s->standard == FZ_STRUCTURE_H ||
		s->standard == FZ_STRUCTURE_H1 ||
		s->standard == FZ_STRUCTURE_H2 ||
		s->standard == FZ_STRUCTURE_H3 ||
		s->standard == FZ_STRUCTURE_H4 ||
		s->standard == FZ_STRUCTURE_H5 ||
		s->standard == FZ_STRUCTURE_H6;
}

static int
is_bullet(int chr)
{
	if (chr == '*' ||
		chr == 0x00B7 || /* Middle Dot */
		chr == 0x2022 || /* Bullet */
		chr == 0x2023 || /* Triangular Bullet */
		chr == 0x2043 || /* Hyphen Bullet */
		chr == 0x204C || /* Back leftwards bullet */
		chr == 0x204D || /* Back rightwards bullet */
		chr == 0x2219 || /* Bullet operator */
		chr == 0x25C9 || /* Fisheye */
		chr == 0x25CB || /* White circle */
		chr == 0x25CF || /* Black circle */
		chr == 0x25D8 || /* Inverse Bullet */
		chr == 0x25E6 || /* White Bullet */
		chr == 0x2619 || /* Reversed Rotated Floral Heart Bullet / Fleuron */
		chr == 0x261a || /* Black left pointing index */
		chr == 0x261b || /* Black right pointing index */
		chr == 0x261c || /* White left pointing index */
		chr == 0x261d || /* White up pointing index */
		chr == 0x261e || /* White right pointing index */
		chr == 0x261f || /* White down pointing index */
		chr == 0x2765 || /* Rotated Heavy Heart Black Heart Bullet */
		chr == 0x2767 || /* Rotated Floral Heart Bullet / Fleuron */
		chr == 0x29BE || /* Circled White Bullet */
		chr == 0x29BF || /* Circled Bullet */
		chr == 0x2660 || /* Black Spade suit */
		chr == 0x2661 || /* White Heart suit */
		chr == 0x2662 || /* White Diamond suit */
		chr == 0x2663 || /* Black Club suit */
		chr == 0x2664 || /* White Spade suit */
		chr == 0x2665 || /* Black Heart suit */
		chr == 0x2666 || /* Black Diamond suit */
		chr == 0x2667 || /* White Clud suit */
		chr == 0x1F446 || /* WHITE UP POINTING BACKHAND INDEX */
		chr == 0x1F447 || /* WHITE DOWN POINTING BACKHAND INDEX */
		chr == 0x1F448 || /* WHITE LEFT POINTING BACKHAND INDEX */
		chr == 0x1F449 || /* WHITE RIGHT POINTING BACKHAND INDEX */
		chr == 0x1f597 || /* White down pointing left hand index */
		chr == 0x1F598 || /* SIDEWAYS WHITE LEFT POINTING INDEX */
		chr == 0x1F599 || /* SIDEWAYS WHITE RIGHT POINTING INDEX */
		chr == 0x1F59A || /* SIDEWAYS BLACK LEFT POINTING INDEX */
		chr == 0x1F59B || /* SIDEWAYS BLACK RIGHT POINTING INDEX */
		chr == 0x1F59C || /* BLACK LEFT POINTING BACKHAND INDEX */
		chr == 0x1F59D || /* BLACK RIGHT POINTING BACKHAND INDEX */
		chr == 0x1F59E || /* SIDEWAYS WHITE UP POINTING INDEX */
		chr == 0x1F59F || /* SIDEWAYS WHITE DOWN POINTING INDEX */
		chr == 0x1F5A0 || /* SIDEWAYS BLACK UP POINTING INDEX */
		chr == 0x1F5A1 || /* SIDEWAYS BLACK DOWN POINTING INDEX */
		chr == 0x1F5A2 || /* BLACK UP POINTING BACKHAND INDEX */
		chr == 0x1F5A3 || /* BLACK DOWN POINTING BACKHAND INDEX */
		chr == 0x1FBC1 || /* LEFT THIRD WHITE RIGHT POINTING INDEX */
		chr == 0x1FBC2 || /* MIDDLE THIRD WHITE RIGHT POINTING INDEX */
		chr == 0x1FBC3 || /* RIGHT THIRD WHITE RIGHT POINTING INDEX */
		0)
		return 1;

	return 0;
}

typedef struct
{
	float space;
	int looking_for_space;
	int maybe_ends_paragraph;
	float line_gap;
	/* Confusingly, we have 2 different previous line gaps stored.
	 * First we have the gap available after the line that was on
	 * a previous physical line. This is the one we really want. */
	float prev_physical_line_gap;
	/* But we don't know when to update that until we find a line
	 * entry that is on a different physical line. So we also have
	 * to keep this, the gap available after the previous line in
	 * the structure (which might actually be on the same physical
	 * line as the line we are looking at now). */
	float prev_line_gap;
	int first_char;
} gather_state;

static void
gather_region_stats_aux(fz_context *ctx, fz_stext_block *block, fz_rect region, fz_features *features, gather_state *state, int is_header)
{
	fz_feature_stats *stats = &features->stats;
	float last_baseline;

	int first_line = 1;
	for (; block != NULL; block = block->next)
	{
		fz_stext_line *line;

		if (block->type == FZ_STEXT_BLOCK_STRUCT)
		{
			if (block->u.s.down != NULL)
			{
				int hdr = is_header || struct_is_header(block->u.s.down);
				gather_region_stats_aux(ctx, block->u.s.down->first_block, region, features, state, hdr);
			}
			continue;
		}
		if (block->type != FZ_STEXT_BLOCK_TEXT)
			continue;
		for (line = block->u.t.first_line; line != NULL; line = line->next)
		{
			fz_stext_char *ch;
			float baseline = line->first_char->origin.y;
			int newline = 1;
			fz_rect clipped_line = fz_intersect_rect(region, line->bbox);

			state->prev_line_gap = state->line_gap;
			state->line_gap = block->bbox.x1 - clipped_line.x1;
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
					int char_is_bullet = is_bullet(ch->c);

					stats->non_line_bullets += char_is_bullet;

					if (stats->is_header == -1)
						stats->is_header = is_header;
					else if (stats->is_header == 2)
					{
						/* Already mixed. */
					}
					else if (stats->is_header != is_header)
					{
						stats->is_header = 2; /* mixed */
					}

					if (!first_line && newline)
					{
						float threshold_line_change = ch->size/4;
						float line_space = baseline - last_baseline;
						if (line_space > threshold_line_change || line_space < -threshold_line_change)
							font_freq_push(ctx, &features->region_linespaces, NULL, line_space);
						newline = 0;
					}
					first_line = 0;
					last_baseline = baseline;

					/* To calculate the indents at top left/bottom right, we need to
					 * look at line y0/y1, not char y0/y1, because otherwise alternating
					 * low and high chars may give confusing results. For example "_'".
					 * The "'" might appear to be in a higher line than the "_". */

					/* If we find a line higher than we've found before (by at least
					 * half the lineheight, to avoid silly effects), recalculate our
					 * top_left_corner. */
					if (line->bbox.y0 < features->top_left_y - features->top_left_height/2)
					{
						features->top_left_y = line->bbox.y0;
						features->top_left_height = line->bbox.y1 - line->bbox.y0;
						stats->top_left_x = char_rect.x0;
					}
					/* Otherwise if we're in the same line, and x is less, take that. */
					else if (line->bbox.y0 < features->top_left_y + features->top_left_height/2 && char_rect.x0 < stats->top_left_x)
					{
						stats->top_left_x = char_rect.x0;
					}

					/* If we find a line lower than we've found before (by at least
					 * half the lineheight, to avoid silly effects)... */
					if (line->bbox.y1 > features->bottom_right_y + features->bottom_right_height/2)
					{
#ifdef DEBUG_CHARS
						fprintf(stderr, "<newline>");
#endif
						/* recalculate our bottom right corner */
						features->bottom_right_y = line->bbox.y1;
						features->bottom_right_height = line->bbox.y1 - line->bbox.y0;
						stats->bottom_right_x = char_rect.x1;
						/* Increment the number of lines, and prepare the indents. */
						stats->num_lines++;
						if (stats->num_lines > 1 && stats->max_non_first_left_indent < line->bbox.x0 - region.x0)
							stats->max_non_first_left_indent = line->bbox.x0 - region.x0;
						if (features->max_right_indent > stats->max_non_last_right_indent)
							stats->max_non_last_right_indent = features->max_right_indent;
						features->max_right_indent = region.x1 - line->bbox.x1;


						state->prev_physical_line_gap = state->prev_line_gap;
						if (state->looking_for_space)
						{
							/* We've moved downwards onto a line, and failed to find
							 * a space on that line. Presumably that means that whole
							 * line is a single word. */
							float line_len = clipped_line.x1 - clipped_line.x0;

							if (line_len + state->space < state->prev_physical_line_gap)
							{
								/* We could have fitted this word into the previous line. */
								stats->dodgy_paragraph_breaks++;
							}
							state->looking_for_space = 0;
						}
#ifdef DEBUG_CHARS
						fprintf(stderr, "<maybe_ends_paragraph=%d>", state->maybe_ends_paragraph);
#endif
						state->looking_for_space = state->maybe_ends_paragraph;

						stats->line_bullets += char_is_bullet;
						stats->non_line_bullets -= char_is_bullet;
						/* HACK: We count an unknown char as a bullet if it's the first thing on
						 * a line. */
						if (ch->c == 0xFFFD)
							stats->line_bullets++;
					}
					/* Otherwise if we're in the same line... */
					else if (line->bbox.y1 < features->bottom_right_y + features->bottom_right_height/2)
					{
						/* if x is greater, take that. */
						if (char_rect.x1 > stats->bottom_right_x)
							stats->bottom_right_x = char_rect.x1;
						/* shrink the right ident we've found so far on this line. */
						features->max_right_indent = region.x1 - line->bbox.x1;
					}

#ifdef DEBUG_CHARS
					fprintf(stderr, "%c", ch->c);
#endif
					/* We have a char to consider */
					if ((ch->c >= '0' && ch->c <= '9') || ch->c == '.' || ch->c == '%')
						stats->num_numerals++;
					else
						stats->num_non_numerals++;

					if ((ch->c >= 'A' && ch->c <= 'Z') ||
						(ch->c >= 'a' && ch->c <= 'z') ||
						(ch->c >= '0' && ch->c <= '9'))
					{
						/* In Latin text, paragraphs should always end up some form
						 * of punctuation. I suspect that's less true of some other
						 * languages (particularly far-eastern ones). Let's just say
						 * that if we end in A-Za-z0-9 we can't possibly be the last
						 * line of a paragraph. */
						state->maybe_ends_paragraph = 0;
#ifdef DEBUG_CHARS
						fprintf(stderr, "0");
#endif
					}
					else
					{
						/* Plausibly the next line might be the first line of a new paragraph */
						state->maybe_ends_paragraph = 1;
#ifdef DEBUG_CHARS
						fprintf(stderr, "1");
#endif
					}

					if (ch->c == 32)
					{
						float this_space = char_rect.x1 - char_rect.x0;
						if (state->space > this_space)
							state->space = this_space;

						if (state->looking_for_space)
						{
							float line_len = char_rect.x0 - line->bbox.x0;
#ifdef DEBUG_CHARS
							fprintf(stderr, "<looking...>");
#endif
							if (line_len + state->space < state->prev_physical_line_gap)
							{
								/* We could have fitted this word into the previous line. */
								stats->dodgy_paragraph_breaks++;
							}
							state->looking_for_space = 0;
						}
					}


					features->num_chars++;
					if (!state->first_char)
					{
						stats->char_space_n++;
						if (stats->last_char_rect.x1 < char_rect.x0)
							stats->char_space += char_rect.x0 - stats->last_char_rect.x1;
					}
					state->first_char = 0;
					stats->last_char_rect = char_rect;

					if (ch->flags & FZ_STEXT_UNDERLINE)
						stats->num_underlines++;

					font_freq_push(ctx, &features->region_fonts, ch->font, ch->size);

					stats->font_size += ch->size;
					features->font_size_n++;

					/* Allow for chars that overlap the edge */
					/* Bit of a hack this. The coords are fed into this program accurate to 3
					 * decimal places. This can round them down. This upsets our margin
					 * calculations, as chars can be seen to extend fractionally outside
					 * the region. Accordingly, we extend the region slightly to avoid this. */
#define ROUND 0.001f
					if (char_rect.x0 < region.x0 - ROUND)
						stats->margin_l = 0;
					if (char_rect.x1 > region.x1 + ROUND)
						stats->margin_r = 0;
					if (char_rect.y0 < region.y0 - ROUND)
						stats->margin_t = 0;
					if (char_rect.y1 > region.y1 + ROUND)
						stats->margin_b = 0;

					/* Inner margins */
					m = char_rect.x0 - region.x0;
					if (m < 0)
						m = 0;
					if (stats->imargin_l > m)
						stats->imargin_l = m;
					m = region.x1 - char_rect.x1;
					if (m < 0)
						m = 0;
					if (stats->imargin_r > m)
						stats->imargin_r = m;
					m = char_rect.y0 - region.y0;
					if (m < 0)
						m = 0;
					if (stats->imargin_t > m)
						stats->imargin_t = m;
					m = region.y1 - char_rect.y1;
					if (m < 0)
						m = 0;
					if (stats->imargin_b > m)
						stats->imargin_b = m;
				}
				else
				{
#ifdef DEBUG_CHARS
					fprintf(stderr, "<%c>", ch->c);
#endif
					if (region.y0 <= p.y && p.y <= region.y1)
					{
						float m = region.x0 - char_rect.x1;
						if (m >= 0 && stats->margin_l > m)
							stats->margin_l = m;
						m = char_rect.x0 - region.x1;
						if (m >= 0 && stats->margin_r > m)
							stats->margin_r = m;
					}
					if (region.x0 <= p.x && p.x <= region.x1)
					{
						float m = region.y0 - char_rect.y1;
						if (m >= 0 && stats->margin_t > m)
							stats->margin_t = m;
						m = char_rect.y0 - region.y1;
						if (m >= 0 && stats->margin_b > m)
							stats->margin_b = m;
					}
				}
			}
#ifdef DEBUG_CHARS
			fprintf(stderr, "\n");
#endif
		}
	}
}

static void
gather_region_stats(fz_context *ctx, fz_stext_block *block, fz_rect region, fz_features *features)
{
	gather_state state = { 0 };

	state.space = 99999;
	state.first_char = 1;

	gather_region_stats_aux(ctx, block, region, features, &state, 0);

	features->stats.num_fonts_in_region = features->region_fonts.len;
}

typedef struct
{
	fz_stext_block *block;
	fz_stext_line *line;
	fz_stext_line *end;
	float metric;
	int is_header;
} best_line;

static void
find_line_above_region(fz_context *ctx, fz_stext_block *block, fz_rect region, best_line *best, int in_header)
{
	for (; block != NULL; block = block->next)
	{
		fz_stext_line *line;

		/* If the whole block is no higher than the region, trivially exclude it. */
		if (block->bbox.y0 >= region.y0)
			continue;
		if (block->type == FZ_STEXT_BLOCK_STRUCT)
		{
			if (block->u.s.down)
			{
				int hdr = in_header || struct_is_header(block->u.s.down);
				find_line_above_region(ctx, block->u.s.down->first_block, region, best, hdr);
			}
			continue;
		}
		else if (block->type != FZ_STEXT_BLOCK_TEXT)
			continue;

		for (line = block->u.t.first_line; line != NULL; line = line->next)
		{
			float score = region.y0 - line->bbox.y1;
			if (score < 0)
				continue;
			if (line->bbox.x0 >= region.x1 || line->bbox.x1 <= region.x0)
				continue;
			if (best->block == NULL || score < best->metric)
			{
				best->block = block;
				best->line = line;
				best->metric = score;
				best->is_header = in_header;
			}
		}
	}
}

static void
find_line_below_region(fz_context *ctx, fz_stext_block *block, fz_rect region, best_line *best, int in_header)
{
	for (; block != NULL; block = block->next)
	{
		fz_stext_line *line;

		/* If the whole block is no lower than the region, trivially exclude it. */
		if (block->bbox.y1 <= region.y1)
			continue;
		if (block->type == FZ_STEXT_BLOCK_STRUCT)
		{
			if (block->u.s.down)
			{
				int hdr = in_header || struct_is_header(block->u.s.down);
				find_line_below_region(ctx, block->u.s.down->first_block, region, best, hdr);
			}
			continue;
		}
		else if (block->type != FZ_STEXT_BLOCK_TEXT)
			continue;

		for (line = block->u.t.first_line; line != NULL; line = line->next)
		{
			float score = line->bbox.y0 - region.y1;
			if (score < 0)
				continue;
			if (line->bbox.x0 >= region.x1 || line->bbox.x1 <= region.x0)
				continue;
			if (best->block == NULL || score < best->metric)
			{
				best->block = block;
				best->line = line;
				best->metric = score;
				best->is_header = in_header;
			}
		}
	}
}

static fz_rect
gather_line(fz_context *ctx, best_line *best, fz_rect region)
{
	fz_stext_line *start = best->line;
	fz_stext_line *end = best->line;
	fz_rect bbox = start->bbox;

	while (start->prev)
	{
		fz_stext_line *prev = start->prev;
		if (prev->bbox.y0 >= best->line->bbox.y1 || prev->bbox.y1 <= best->line->bbox.y0)
			break;
		//if (prev->bbox.x1 < region.x0 || prev->bbox.x0 > region.x1 )
		//	break;
		start = prev;
		bbox = fz_union_rect(bbox, start->bbox);
	}
	while (end->next)
	{
		fz_stext_line *next = end->next;
		if (next->bbox.y0 >= best->line->bbox.y1 || next->bbox.y1 <= best->line->bbox.y0)
			break;
		//if (next->bbox.x1 < region.x0 || next->bbox.x0 > region.x1 )
		//	break;
		end = next;
		bbox = fz_union_rect(bbox, end->bbox);
	}

	best->line = start;
	best->end = end;

	return bbox;
}

static float
average_font_size(fz_context *ctx, fz_stext_line *start, fz_stext_line *end)
{
	int n = 0;
	float sum = 0;

	while (1)
	{
		fz_stext_char *ch;

		for (ch = start->first_char; ch != NULL; ch = ch->next)
		{
			n++;
			sum += ch->size;
		}
		if (start == end)
			break;
		start = start->next;
	}

	return n == 0 ? 12.0f : (sum/n);
}

static void
contextual_features_above(fz_context *ctx, fz_stext_block *block, fz_rect region, fz_feature_stats *stats)
{
	best_line best = { 0 };
	fz_rect bbox;
	float size, delta;

	find_line_above_region(ctx, block, region, &best, 0);
	if (best.block == NULL)
	{
		/* There is no line above the region. */
		return;
	}
	bbox = gather_line(ctx, &best, region);

	if (bbox.y1 + MAX_MARGIN <= region.y0)
	{
		/* This line is outside margin range. */
		return;
	}

	size = average_font_size(ctx, best.line, best.end);
	delta = fabsf(size - stats->font_size);
	
	stats->context_above_font_size = stats->font_size != 0 ? delta / stats->font_size : 0;
	stats->context_above_is_header = best.is_header;
	stats->context_above_outdent = region.x1 - bbox.x1;
	stats->context_above_indent = bbox.x0 - region.x0;
}

static void
contextual_features_below(fz_context *ctx, fz_stext_block *block, fz_rect region, fz_feature_stats *stats)
{
	best_line best = { 0 };
	fz_rect bbox;
	float size, delta;

	find_line_below_region(ctx, block, region, &best, 0);
	if (best.block == NULL)
	{
		/* There is no line below the region. */
		return;
	}
	bbox = gather_line(ctx, &best, region);

	if (bbox.y0 - MAX_MARGIN >= region.y1)
	{
		/* This line is outside margin range. */
		return;
	}
	size = average_font_size(ctx, best.line, best.end);
	delta = fabsf(size - stats->font_size);
	
	stats->context_below_font_size = stats->font_size != 0 ? delta / stats->font_size : 0;
	stats->context_below_is_header = best.is_header;
	stats->context_below_outdent = bbox.x1 - stats->bottom_right_x;
	stats->context_below_indent = bbox.x0 - region.x0;
	stats->context_below_bullet = (best.line && best.line->first_char ? is_bullet(best.line->first_char->c) : 0);
}

static void
process_global_font_stats(fz_context *ctx, fz_rect region, fz_features *features)
{
	fz_feature_stats *stats = &features->stats;

	if (stats->char_space_n)
		stats->char_space /= stats->char_space_n;

	stats->ratio = features->num_chars / ((region.x1-region.x0) * (region.y1-region.y0));
	if (features->font_size_n)
		stats->font_size /= features->font_size_n;

	font_freq_common(ctx, &features->fonts, 0.5);
	font_freq_common(ctx, &features->linespaces, 0.5);

	features->fonts_mode = font_freq_mode(ctx, &features->fonts);
	features->linespaces_mode = font_freq_mode(ctx, &features->linespaces);
}

static void
process_region_font_stats(fz_context *ctx, fz_rect region, fz_features *features)
{
	fz_feature_stats *stats = &features->stats;

	font_freq_common(ctx, &features->region_fonts, 0.5);
	font_freq_common(ctx, &features->region_linespaces, 0.5);

	stats->fonts_offset = 0;
	if (features->fonts_mode != -1)
	{
		features->rfonts_mode = font_freq_mode(ctx, &features->region_fonts);
		if (features->rfonts_mode != -1)
			stats->fonts_offset = font_freq_closest(ctx, &features->fonts, features->region_fonts.list[features->rfonts_mode].font, features->region_fonts.list[features->rfonts_mode].size) - features->fonts_mode;
	}
	stats->linespaces_offset = 0;
	stats->line_space = 0;
	if (features->linespaces_mode != -1)
	{
		features->rlinespaces_mode = font_freq_mode(ctx, &features->region_linespaces);
		if (features->rlinespaces_mode != -1)
		{
			stats->line_space = features->region_linespaces.list[features->rlinespaces_mode].size;
			stats->linespaces_offset = font_freq_closest(ctx, &features->linespaces, NULL, stats->line_space);
			stats->linespaces_offset -= features->linespaces_mode;
		}
	}

	/* Horrible, but will turn region_fonts into a list of the unique fonts in this region. */
	font_freq_common(ctx, &features->region_fonts, 100000);
}

fz_features *
fz_new_page_features(fz_context *ctx, fz_stext_page *page)
{
	fz_features *features = NULL;

	fz_var(features);

	fz_try(ctx)
	{
		features = fz_malloc_struct(ctx, fz_features);
		features->page = page;
		page = NULL;
	}
	fz_always(ctx)
	{
		fz_drop_stext_page(ctx, page);
	}
	fz_catch(ctx)
	{
		fz_rethrow(ctx);
	}

	/* Collect global stats */
	gather_global_stats(ctx, features->page->first_block, features);
	process_global_font_stats(ctx, features->page->mediabox, features);

	return features;
}

void
fz_drop_page_features(fz_context *ctx, fz_features *features)
{
	if (features == NULL)
		return;

	font_freq_drop(ctx, &features->fonts);
	font_freq_drop(ctx, &features->region_fonts);
	font_freq_drop(ctx, &features->linespaces);
	font_freq_drop(ctx, &features->region_linespaces);

	fz_drop_stext_page(ctx, features->page);

	fz_free(ctx, features);
}

fz_feature_stats *
fz_features_for_region(fz_context *ctx, fz_features *features, fz_rect region, int category)
{
	fz_feature_stats *stats = &features->stats;
	fz_stext_page *page = features->page;
	stats->margin_l = MAX_MARGIN;
	stats->margin_r = MAX_MARGIN;
	stats->margin_t = MAX_MARGIN;
	stats->margin_b = MAX_MARGIN;
	stats->imargin_l = MAX_MARGIN;
	stats->imargin_r = MAX_MARGIN;
	stats->imargin_t = MAX_MARGIN;
	stats->imargin_b = MAX_MARGIN;
	stats->top_left_x = region.x1;
	features->top_left_y = region.y1;
	stats->bottom_right_x = region.x0;
	features->bottom_right_y = region.y0;
	stats->is_header = -1;

	/* Now collect for the region itself. */
	gather_region_stats(ctx, page->first_block, region, features);
	process_region_font_stats(ctx, region, features);

	/* Now we scan again, to try to make contextual features. */
	contextual_features_above(ctx, page->first_block, region, &features->stats);
	contextual_features_below(ctx, page->first_block, region, &features->stats);
	stats->context_header_differs = stats->is_header != stats->context_above_is_header || stats->is_header != stats->context_below_is_header;

	stats->top_left_x -= region.x0;
	if (stats->top_left_x < 0)
		stats->top_left_x = 0;
	stats->bottom_right_x = region.x1 - stats->bottom_right_x;
	if (stats->bottom_right_x < 0)
		stats->bottom_right_x = 0;

	/* Calculate some synthetic features */
	stats->smargin_l = stats->margin_l / (stats->font_size != 0 ? stats->font_size : 12);
	stats->smargin_r = stats->margin_r / (stats->font_size != 0 ? stats->font_size : 12);
	stats->smargin_t = stats->margin_t / (stats->line_space != 0 ? fabs(stats->line_space) : 1.2f * (stats->font_size != 0 ? stats->font_size : 12));
	stats->smargin_b = stats->margin_b / (stats->line_space != 0 ? fabs(stats->line_space) : 1.2f * (stats->font_size != 0 ? stats->font_size : 12));

	return stats;
}

