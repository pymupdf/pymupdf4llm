// Copyright (C) 2024 Artifex Software, Inc.
//
// This software may only be used under license.

#include "utils.h"

#ifdef _WIN32
#define DIR_SEP '\\'
#else
#define DIR_SEP '/'
#endif

char *
make_prefixed_name(fz_context *ctx, const char *directory, char *filename)
{
	size_t z = 0;
	int add_sep = 0;
	char *buf;

	if (directory != NULL)
	{
		z = strlen(directory);
		if (z == 0 || directory[z-1] != DIR_SEP)
			add_sep = 1;
	}
	buf = fz_malloc(ctx, z + add_sep + strlen(filename) + 1);
	if (directory)
		memcpy(buf, directory, z);
	if (add_sep)
		buf[z] = DIR_SEP;
	strcpy(&buf[z+add_sep], filename);

	return buf;
}
