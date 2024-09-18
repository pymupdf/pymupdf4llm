// Copyright (C) 2024 Artifex Software, Inc.
//
// This software may only be used under license.

#ifndef UTILS_H
#define UTILS_H

#include "mupdf/fitz.h"

char *make_prefixed_name(fz_context *ctx, const char *directory, char *filename);

#endif /* UTILS_H */
