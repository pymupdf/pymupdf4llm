// Copyright (C) 2024 Artifex Software, Inc.
//
// This software may only be used under license.

// features.h is a special header with gcc.
#include "features_decls.h"

#include "csv.h"
#include "utils.h"

#include <stdio.h>

/* #define DEBUG_CHARS */
/* #define DEBUG_RAFT */
/* #define DEBUG_RAFT_AS_PS */

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

/* A bunch of rafts makes a flotilla. */
typedef struct
{
	fz_rect area;
} raft_t;

typedef struct
{
	int len;
	int max;
	raft_t *rafts;
} flotilla_t;

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
	int char_space_n;
	int fonts_mode;
	int linespaces_mode;
	int rfonts_mode;
	int rlinespaces_mode;
	float top_left_y;
	float top_left_height;
	float bottom_right_y;
	float bottom_right_height;
	float max_right_indent;
	/* Calculated during the first pass, the distance to the nearest
	 * non-aligned block. */
	float nearest_nonaligned_up;
	float nearest_nonaligned_down;
	float nearest_nonaligned_left;
	float nearest_nonaligned_right;
	flotilla_t *vector_flotilla;
	flotilla_t *image_flotilla;
};

#if 1
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

/*
	Raft handling

	We call any 2-dimensional area that's covered by (some type of) content
	a raft. i.e. it's made up of several distinct objects lashed together
	into something that covers a large flat area.

	For instance, the borders and/or backgrounds from a table would form a
	raft behind the text content. And the boundaries of that raft might
	help us distinguish that table from an adjacent table on a different
	raft.

	While we could theorectically make rafts from anything, images and
	vectors seem like the best bet. We could make rafts from mixed images
	and vectors, but to start with, I think we'll get best results from
	images and vectors separately.
*/
static flotilla_t *
new_flotilla(fz_context *ctx)
{
	return fz_malloc_struct(ctx, flotilla_t);
}

static void
drop_flotilla(fz_context *ctx, flotilla_t *f)
{
	if (!f)
		return;
	fz_free(ctx, f->rafts);
	fz_free(ctx, f);
}

static void
add_raft_to_flotilla(fz_context *ctx, flotilla_t *f, raft_t r)
{
	if (f->len == f->max)
	{
		int newmax = f->max * 2;
		if (newmax == 0)
			newmax = 8;
		f->rafts = (raft_t *) fz_realloc(ctx, f->rafts, sizeof(f->rafts[0]) * newmax);
		f->max = newmax;
	}

	f->rafts[f->len++] = r;
}

/* Without FUDGE this would be equivalent to:
 * fz_is_valid_rect(fz_intersect_rect(r, s))
 * but faster.
 *
 * With FUDGE it is slightly forgiving to allow for FP inaccuracies.
 */
 #define FUDGE 0.1
static int
overlap_or_abut(fz_rect r, fz_rect s)
{
	return (r.x1 >= s.x0 - FUDGE && r.x0 <= s.x1 + FUDGE && r.y1 >= s.y0 - FUDGE && r.y0 <= s.y1 + FUDGE);
}

#ifdef DEBUG_RAFT
static void
verify(fz_context *ctx, flotilla_t *f)
{
	int i, j;

	printf("Dump: len=%d\n", f->len);
	for (i = 0; i < f->len; i++)
	{
		printf("%d: %g %g %g %g\n", i, f->rafts[i].area.x0, f->rafts[i].area.y0, f->rafts[i].area.x1, f->rafts[i].area.y1);
	}

	for (i = 0; i < f->len-1; i++)
	{
		for (j = i+1; j < f->len; j++)
		{
			if (overlap_or_abut(f->rafts[i].area, f->rafts[j].area))
			{
				printf("%d and %d overlap!\n", i, j);
				assert(i == j);
			}
		}
	}
}
#endif

static void
add_plank_to_flotilla(fz_context *ctx, flotilla_t *f, fz_rect rect)
{
	int i, j;
	int overlaps = -1;

#ifdef DEBUG_RAFT
	verify(ctx, f);

	printf("%g %g %g %g\n", rect.x0, rect.y0, rect.x1, rect.y1);
#endif

	/* Does the plank extend any of the existing rafts? */
	for (i = f->len-1; i >= 0; i--)
	{
		if (overlap_or_abut(rect, f->rafts[i].area))
		{
			/* We overlap. */
			fz_rect r = fz_union_rect(f->rafts[i].area, rect);
			if (r.x0 == f->rafts[i].area.x0 &&
				r.y0 == f->rafts[i].area.y0 &&
				r.x1 == f->rafts[i].area.x1 &&
				r.y1 == f->rafts[i].area.y1)
			{
				/* We were entirely contained. Nothing more to do. */
#ifdef DEBUG_RAFT
				printf("Contained\n");
#endif
				return;
			}
			f->rafts[i].area = r;
#ifdef DEBUG_RAFT
			printf("Overlap %d -> %g %g %g %g\n", i, r.x0, r.y0, r.x1, r.y1);
#endif
			break;
		}
	}

	if (i >= 0)
	{
		/* We've extended raft[i]. We now need to check if any other raft overlaps
		 * with the extended one. */

		/* But our new one might have bridged between two (or more!) existing rafts. */
		/* Unfortunately, if we bridge between 2 rafts, that new larger raft might now
		 * intersect with more rafts. So we need to repeatedly scan. */
		while (1)
		{
			int changed = 0;

			for (j = f->len-1; j > i; j--)
			{
				if (overlap_or_abut(f->rafts[j].area, f->rafts[i].area))
				{
					/* Update raft i to be the union of the two. */
					f->rafts[i].area = fz_union_rect(f->rafts[j].area, f->rafts[i].area);
#ifdef DEBUG_RAFT
					printf("Bridge %d -> %g %g %g %g\n", j, f->rafts[i].area.x0, f->rafts[i].area.y0, f->rafts[i].area.x1, f->rafts[i].area.y1);
#endif
					/* Shorten the list by moving the end one down to be the ith one. */
					f->len--;
					if (j != f->len)
					{
						f->rafts[j] = f->rafts[f->len];
					}
					changed = 1;
				}
			}

			for (j = i-1; j >= 0; j--)
			{
				if (overlap_or_abut(f->rafts[j].area, f->rafts[i].area))
				{
					/* Update raft j to be the union of the two. */
					f->rafts[j].area = fz_union_rect(f->rafts[j].area, f->rafts[i].area);
#ifdef DEBUG_RAFT
					printf("Bridge %d -> %g %g %g %g\n", j, f->rafts[j].area.x0, f->rafts[j].area.y0, f->rafts[j].area.x1, f->rafts[j].area.y1);
#endif
					/* Shorten the list by moving the end one down to be the ith one. */
					f->len--;
					if (i != f->len)
					{
						f->rafts[i] = f->rafts[f->len];
						i = j;
					}
					changed = 1;
				}
			}

			if (!changed)
				break;
		}
	}
	else
	{
		/* This didn't overlap anything else. Make a new raft. */
		raft_t raft = { rect };
		add_raft_to_flotilla(ctx, f, raft);
	}
}

static int
find_raft(flotilla_t *flotilla, fz_rect region)
{
	int i;

	for (i = 0; i < flotilla->len; i++)
	{
		if (fz_contains_rect(flotilla->rafts[i].area, region))
			return i;
	}
	return -1;
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
		if (block->type == FZ_STEXT_BLOCK_VECTOR)
		{
			add_plank_to_flotilla(ctx, features->vector_flotilla, block->bbox);
		}
		if (block->type != FZ_STEXT_BLOCK_TEXT)
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
	int have_baseline;
	float baseline;
	float baseline_fontsize;
} gather_state;

static void
gather_region_stats_aux(fz_context *ctx, fz_stext_block *block, fz_rect region, fz_features *features, gather_state *state, int is_header)
{
	fz_feature_stats *stats = &features->stats;
	float last_baseline;
	int first_line = 1;

	assert(is_header == 0 || is_header == 1 || is_header == 2);

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

			if (fz_contains_rect(region, line->bbox))
			{
				if (!(line->dir.y == 0 && line->dir.x != 0))
				{
					/* Not a horizontal line. Don't mess with baselines. */
				}
				else if (!state->have_baseline)
				{
					/* This is the first time we've found a line completely contained within
					 * the region. So remember the baseline. */
					state->have_baseline = 1;
					state->baseline = baseline;
					state->baseline_fontsize = line->first_char->size;
				}
				else if (state->baseline_fontsize > line->first_char->size)
				{
					/* Ignore this line because the font size is smaller than the
					 * one we found before. This avoids us tripping on sub or
					 * super-scripts. */
				}
				else if (state->baseline_fontsize < line->first_char->size)
				{
					/* Take a larger fontsize in preference. */
					state->baseline = baseline;
					state->baseline_fontsize = line->first_char->size;
				}
				else if (state->baseline > baseline)
				{
					/* Take a higher baseline in preference. So we take the baseline of
					 * the first line of a paragraph in the region rather than a later
					 * one. */
					state->baseline = baseline;
					state->baseline_fontsize = line->first_char->size;
				}
			}

			/* In the first pass, we find the distance to the first line that
			 * is not aligned in at least some way in each direction. */
			if (line->bbox.x1 <= region.x0 || line->bbox.x0 >= region.x1)
			{
				/* Line does not overlap horizontally, so we can ignore it. */
			}
			else if (feq(line->bbox.x0, region.x0) ||
				feq(line->bbox.x1, region.x1) ||
				feq(line->bbox.x0+line->bbox.x1, region.x0+region.x1))
			{
				/* Aligned in at least some way. */
			}
			else
			{
				float d = region.y0 - line->bbox.y1;
				if (d > 0 && d < features->nearest_nonaligned_up)
					features->nearest_nonaligned_up = d;
				d =  line->bbox.y0 - region.y1;
				if (d > 0 && d < features->nearest_nonaligned_down)
					features->nearest_nonaligned_down = d;
			}

			if (line->bbox.y1 <= region.y0 || line->bbox.y0 >= region.y1)
			{
				/* Line does not overlap horizontally, so we can ignore it. */
			}
			else if (feq(line->bbox.y0, region.y0) ||
				feq(line->bbox.y1, region.y1) ||
				feq(line->bbox.y0+line->bbox.y1, region.y0+region.y1))
			{
				/* Aligned in at least some way. */
			}
			else
			{
				float d = region.x0 - line->bbox.x1;
				if (d > 0 && d < features->nearest_nonaligned_left)
					features->nearest_nonaligned_left = d;
				d =  line->bbox.x0 - region.x1;
				if (d > 0 && d < features->nearest_nonaligned_right)
					features->nearest_nonaligned_right = d;
			}

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
						features->char_space_n++;
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
gather_region_stats2_aux(fz_context *ctx, fz_stext_block *block, fz_rect region, fz_features *features, gather_state *state)
{
	fz_feature_stats *stats = &features->stats;

	for (; block != NULL; block = block->next)
	{
		fz_stext_line *line;

		if (block->type == FZ_STEXT_BLOCK_STRUCT)
		{
			if (block->u.s.down != NULL)
			{
				gather_region_stats2_aux(ctx, block->u.s.down->first_block, region, features, state);
			}
			continue;
		}
		if (block->type == FZ_STEXT_BLOCK_VECTOR)
		{
			/* We are only interested in rectangles. */
			float d;
			if ((block->u.v.flags & FZ_STEXT_VECTOR_IS_RECTANGLE) == 0)
				continue;

			/* We're only interested in blocks that cover the centre of our bbox */
			d = (region.y0 + region.y1)/2;
			if (block->bbox.y0 <= d && block->bbox.y1 >= d)
			{
				/* 2 cases here. Firstly, does this rectangle cover us? */
				if (block->bbox.x0 <= region.x0 && block->bbox.x1 >= region.x1)
				{
					/* Covered */
					d = region.x0 - block->bbox.x0;
					if (d >= 0 && d < features->stats.margin_l && features->stats.ray_line_distance_left > d)
						features->stats.ray_line_distance_left = d;
					d = block->bbox.x1 - region.x1;
					if (d >= 0 && d < features->stats.margin_r && features->stats.ray_line_distance_right > d)
						features->stats.ray_line_distance_right = d;
				}
				else
				{
					/* We're looking for a block off to one side of us. */
					d = region.x0 - block->bbox.x1;
					if (d >= 0 && d < features->stats.margin_l && features->stats.ray_line_distance_left > d)
						features->stats.ray_line_distance_left = d;
					d = block->bbox.x0 - region.x1;
					if (d >= 0 && d < features->stats.margin_r && features->stats.ray_line_distance_right > d)
						features->stats.ray_line_distance_right = d;
				}
			}
			d = (region.x0 + region.x1)/2;
			if (block->bbox.x0 <= d && block->bbox.x1 >= d)
			{
				/* 2 cases here. Firstly, does this rectangle cover us? */
				if (block->bbox.y0 <= region.y0 && block->bbox.y1 >= region.y1)
				{
					/* Covered */
					d = region.y0 - block->bbox.y0;
					if (d >= 0 && d < features->stats.margin_t && features->stats.ray_line_distance_up > d)
						features->stats.ray_line_distance_up = d;
					d = block->bbox.y1 - region.y1;
					if (d >= 0 && d < features->stats.margin_b && features->stats.ray_line_distance_down > d)
						features->stats.ray_line_distance_down = d;
				}
				else
				{
					/* We're looking for a block off to one side of us. */
					d = region.y0 - block->bbox.y1;
					if (d >= 0 && d < features->stats.margin_t && features->stats.ray_line_distance_up > d)
						features->stats.ray_line_distance_up = d;
					d = block->bbox.y0 - region.y1;
					if (d >= 0 && d < features->stats.margin_b && features->stats.ray_line_distance_down > d)
						features->stats.ray_line_distance_down = d;
				}
			}
			continue;
		}
		if (block->type != FZ_STEXT_BLOCK_TEXT)
			continue;
		for (line = block->u.t.first_line; line != NULL; line = line->next)
		{
			float d;

			/* In the second pass, we can count the number of blocks that are aligned
			 * in at least some way within the previously found limits */
			d = region.y0 - line->bbox.y1;
			if (d > 0 && d < features->nearest_nonaligned_up)
			{
				if (feq(line->bbox.x0, region.x0))
					features->stats.consecutive_left_alignment_count_up++;
				if (feq(line->bbox.x1, region.x1))
					features->stats.consecutive_right_alignment_count_up++;
				if (feq(line->bbox.x0+line->bbox.x1, region.x0+region.x1))
					features->stats.consecutive_centre_alignment_count_up++;
			}
			d = line->bbox.y0 - region.y1;
			if (d > 0 && d < features->nearest_nonaligned_down)
			{
				if (feq(line->bbox.x0, region.x0))
					features->stats.consecutive_left_alignment_count_down++;
				if (feq(line->bbox.x1, region.x1))
					features->stats.consecutive_right_alignment_count_down++;
				if (feq(line->bbox.x0+line->bbox.x1, region.x0+region.x1))
					features->stats.consecutive_centre_alignment_count_down++;
			}
			d = region.x0 - line->bbox.x1;
			if (d > 0 && d < features->nearest_nonaligned_left)
			{
				if (feq(line->bbox.y0, region.y0))
					features->stats.consecutive_top_alignment_count_left++;
				if (feq(line->bbox.y1, region.y1))
					features->stats.consecutive_bottom_alignment_count_left++;
				if (feq(line->bbox.y0+line->bbox.y1, region.y0+region.y1))
					features->stats.consecutive_middle_alignment_count_left++;
				if (state->have_baseline && feq(line->first_char->origin.y, state->baseline))
					features->stats.consecutive_baseline_alignment_count_left++;
			}
			d = line->bbox.x0 - region.x1;
			if (d > 0 && d < features->nearest_nonaligned_right)
			{
				if (feq(line->bbox.y0, region.y0))
					features->stats.consecutive_top_alignment_count_right++;
				if (feq(line->bbox.y1, region.y1))
					features->stats.consecutive_bottom_alignment_count_right++;
				if (feq(line->bbox.y0+line->bbox.y1, region.y0+region.y1))
					features->stats.consecutive_middle_alignment_count_right++;
				if (state->have_baseline && feq(line->first_char->origin.y, state->baseline))
					features->stats.consecutive_baseline_alignment_count_right++;
			}
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

	if (features->stats.is_header == -1)
		features->stats.is_header = 0;

	gather_region_stats2_aux(ctx, block, region, features, &state);

	features->stats.num_fonts_in_region = features->region_fonts.len;

	features->stats.alignment_up_with_left = (features->stats.consecutive_left_alignment_count_up >= 1);
	features->stats.alignment_up_with_centre = (features->stats.consecutive_centre_alignment_count_up >= 1);
	features->stats.alignment_up_with_right = (features->stats.consecutive_right_alignment_count_up >= 1);
	features->stats.alignment_down_with_left = (features->stats.consecutive_left_alignment_count_down >= 1);
	features->stats.alignment_down_with_centre = (features->stats.consecutive_centre_alignment_count_down >= 1);
	features->stats.alignment_down_with_right = (features->stats.consecutive_right_alignment_count_down >= 1);
	features->stats.alignment_left_with_top = (features->stats.consecutive_top_alignment_count_left >= 1);
	features->stats.alignment_left_with_middle = (features->stats.consecutive_middle_alignment_count_left >= 1);
	features->stats.alignment_left_with_baseline = (features->stats.consecutive_baseline_alignment_count_left >= 1);
	features->stats.alignment_left_with_bottom = (features->stats.consecutive_bottom_alignment_count_left >= 1);
	features->stats.alignment_right_with_top = (features->stats.consecutive_top_alignment_count_right >= 1);
	features->stats.alignment_right_with_middle = (features->stats.consecutive_middle_alignment_count_right >= 1);
	features->stats.alignment_right_with_baseline = (features->stats.consecutive_baseline_alignment_count_right >= 1);
	features->stats.alignment_right_with_bottom = (features->stats.consecutive_bottom_alignment_count_right >= 1);
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

	font_freq_common(ctx, &features->fonts, 0.5);
	font_freq_common(ctx, &features->linespaces, 0.5);

	features->fonts_mode = font_freq_mode(ctx, &features->fonts);
	features->linespaces_mode = font_freq_mode(ctx, &features->linespaces);
}

static void
process_region_font_stats(fz_context *ctx, fz_rect region, fz_features *features)
{
	fz_feature_stats *stats = &features->stats;

	{
		float area = ((region.x1-region.x0) * (region.y1-region.y0));
		/* Using 1 for area being empty is slightly hacky, but the most
		 * common case for this is for a region that just contains a single
		 * 'space' character (with 0 height), so 1 is the right answer. */
		stats->ratio = (area < 0.001) ? 1 : (features->num_chars / area);
	}

	font_freq_common(ctx, &features->region_fonts, 0.5);
	font_freq_common(ctx, &features->region_linespaces, 0.5);

	if (features->font_size_n)
		stats->font_size /= features->font_size_n;
	if (features->char_space_n)
		stats->char_space /= features->char_space_n;
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

static void
process_raft_features(fz_context *ctx, fz_rect region, fz_features *features)
{
	int raft = find_raft(features->vector_flotilla, region);

	features->stats.raft_num = raft+1;
	if (raft == -1)
	{
		features->stats.raft_edge_up = 0;
		features->stats.raft_edge_down = 0;
		features->stats.raft_edge_left = 0;
		features->stats.raft_edge_right = 0;
	}
	else
	{
		fz_rect a = features->vector_flotilla->rafts[raft].area;
		features->stats.raft_edge_up = region.y0 - a.y0;
		features->stats.raft_edge_down = a.y1 - region.y1;
		features->stats.raft_edge_left = region.x0 - a.x0;
		features->stats.raft_edge_right = a.x1 - region.x1;
	}
}

#ifdef DEBUG_RAFT_AS_PS
static void
dump_flotilla_as_ps(fz_context *ctx, flotilla_t *f, int col)
{
	int i;

	fz_write_printf(ctx, fz_stddbg(ctx), "%g %g %g setrgbcolor\n0 setlinewidth\n",
		(col & 0xFF)/255.0f,
		((col>>8) & 0xFF)/255.0f,
		((col>>16) & 0xFF)/255.0f);

	for (i = 0; i < f->len; i++)
	{
		fz_rect *r = &f->rafts[i].area;
		fz_write_printf(ctx, fz_stddbg(ctx), "%g %g moveto %g %g lineto %g %g lineto %g %g lineto closepath stroke\n",
			r->x0, r->y0, r->x0, r->y1, r->x1, r->y1, r->x1, r->y0);
	}

	fz_write_printf(ctx, fz_stddbg(ctx), "showpage\n");
}
#endif


#if FZ_VERSION_MAJOR > 1 || (FZ_VERSION_MAJOR == 1 && FZ_VERSION_MINOR >= 27)
    #define STEXT_PAGE_IS_REFERENCE_COUNTED
#endif

fz_features *
fz_new_page_features(fz_context *ctx, fz_stext_page *page)
{
	fz_features *features = NULL;

	fz_var(features);

	features = fz_malloc_struct(ctx, fz_features);
	#ifdef STEXT_PAGE_IS_REFERENCE_COUNTED
		features->page = fz_keep_stext_page(ctx, page);
	#else
		features->page = page;
	#endif
	fz_try(ctx)
	{
		features->vector_flotilla = new_flotilla(ctx);
		features->image_flotilla = new_flotilla(ctx);
		/* Collect global stats */
		gather_global_stats(ctx, features->page->first_block, features);
		process_global_font_stats(ctx, features->page->mediabox, features);
#ifdef DEBUG_RAFT_AS_PS
		dump_flotilla_as_ps(ctx, features->vector_flotilla, 0);
#endif

	}
	fz_catch(ctx)
	{
		fz_drop_page_features(ctx, features);
		fz_rethrow(ctx);
	}
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
	#ifdef STEXT_PAGE_IS_REFERENCE_COUNTED
		fz_drop_stext_page(ctx, features->page);
	#endif
	drop_flotilla(ctx, features->vector_flotilla);
	drop_flotilla(ctx, features->image_flotilla);

	fz_free(ctx, features);
}

fz_feature_stats *
fz_features_for_region(fz_context *ctx, fz_features *features, fz_rect region, int category)
{
	fz_feature_stats *stats = &features->stats;
	fz_stext_page *page = features->page;
	float max_h = page->mediabox.y1 - page->mediabox.y0;
	float max_w = page->mediabox.x1 - page->mediabox.x0;
	memset(stats, 0, sizeof(*stats));
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
	stats->ray_line_distance_up = max_h;
	stats->ray_line_distance_down = max_h;
	stats->ray_line_distance_left = max_w;
	stats->ray_line_distance_right = max_w;
	features->nearest_nonaligned_up = max_h;
	features->nearest_nonaligned_down = max_h;
	features->nearest_nonaligned_left = max_w;
	features->nearest_nonaligned_right = max_w;
	features->font_size_n = 0;
	features->char_space_n = 0;

	/* Now collect for the region itself. */
	gather_region_stats(ctx, page->first_block, region, features);
	process_region_font_stats(ctx, region, features);
	process_raft_features(ctx, region, features);

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

