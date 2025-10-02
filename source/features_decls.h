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
