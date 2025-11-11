// Copyright (C) 2024 Artifex Software, Inc.
//
// This software may only be used under license.

#ifndef MUPDF_FITZ_FEATURES_H
#define MUPDF_FITZ_FEATURES_H

#include "mupdf/fitz.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
	int num_non_numerals;
	int num_numerals;
	float ratio;
	float char_space;
	float line_space;
	float font_size;
	int num_fonts_in_region;
	float margin_l;
	float margin_r;
	float margin_t;
	float margin_b;
	float imargin_l;
	float imargin_r;
	float imargin_t;
	float imargin_b;
	int fonts_offset;
	int linespaces_offset;
	int num_underlines;
	float top_left_x;
	float bottom_right_x;
	int num_lines;
	float max_non_first_left_indent;
	float max_non_last_right_indent;
	float smargin_l;
	float smargin_t;
	float smargin_r;
	float smargin_b;
	int dodgy_paragraph_breaks;
	int char_space_n;
	fz_rect last_char_rect;
	int is_header;
	float context_above_font_size;
	int context_above_is_header;
	float context_above_indent;
	float context_above_outdent;
	float context_below_font_size;
	int context_below_is_header;
	float context_below_indent;
	float context_below_outdent;
	int context_below_bullet;
	int context_header_differs;
	int line_bullets;
	int non_line_bullets;
	/* How many regions have we found above us that match the alignment of this
	 * block in each of 3 ways, before we mismatch alignment with a text line? */
	int consecutive_left_alignment_count_up;
	int consecutive_centre_alignment_count_up;
	int consecutive_right_alignment_count_up;
	int consecutive_centre_alignment_count_down;
	int consecutive_left_alignment_count_down;
	int consecutive_right_alignment_count_down;
	int consecutive_top_alignment_count_left;
	int consecutive_middle_alignment_count_left;
	int consecutive_bottom_alignment_count_left;
	int consecutive_top_alignment_count_right;
	int consecutive_middle_alignment_count_right;
	int consecutive_bottom_alignment_count_right;
	/* Boolean versions of the above */
	int alignment_up_with_left;
	int alignment_up_with_centre;
	int alignment_up_with_right;
	int alignment_down_with_left;
	int alignment_down_with_centre;
	int alignment_down_with_right;
	int alignment_left_with_top;
	int alignment_left_with_middle;
	int alignment_left_with_bottom;
	int alignment_right_with_top;
	int alignment_right_with_middle;
	int alignment_right_with_bottom;
	/* Distance to closest plausible bordering line. */
	float ray_line_distance_up;
	float ray_line_distance_down;
	float ray_line_distance_left;
	float ray_line_distance_right;
	int raft_num;
	float raft_edge_up;
	float raft_edge_down;
	float raft_edge_left;
	float raft_edge_right;
} fz_feature_stats;

typedef struct fz_features fz_features;

/* If mupdf>=1.27 we use fz_keep_stext_page() and
fz_drop_stext_page(). Otherwise it is caller's responsibility to keep <page>
alive for the duration of the returned fz_features. */

fz_features *fz_new_page_features(fz_context *ctx, fz_stext_page *page);

void fz_drop_page_features(fz_context *ctx, fz_features *features);

fz_feature_stats *fz_features_for_region(fz_context *ctx, fz_features *features, fz_rect region, int category);

#ifdef __cplusplus
}
#endif

#endif /* MUPDF_FITZ_FEATURES_H */
