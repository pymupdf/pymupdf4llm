// Copyright (C) 2024 Artifex Software, Inc.
//
// This software may only be used under license.

#include "csv.h"

static char csv_read_line_buf[4096];
static char *csv_read_line_ptrs[32];

char **
csv_read_line(FILE *file, int *pn)
{
	int c, len, n;

	len = 0;
	n = 1;

	c = fgetc(file);
	if (c == 10 || c == 13)
		c = fgetc(file);
	if (c == EOF)
	{
		*pn = 0;
		return NULL;
	}

	csv_read_line_ptrs[0] = &csv_read_line_buf[len];
	do
	{
		csv_read_line_buf[len++] = c;
		c = fgetc(file);
		if (c == ',')
		{
			c = 0;
			csv_read_line_ptrs[n++] = &csv_read_line_buf[len+1];
		}	
	}
	while (c != EOF && c != 13 && c != 10);

	csv_read_line_buf[len] = 0;
	if (len == 0)
		n = 0;
	*pn = n;

	return csv_read_line_ptrs;
}
