%{

#include <stdio.h>

#include "features_decls.h"

int features_test(const char* t)
{
    fz_context* ctx = fz_new_context(nullptr, nullptr, FZ_STORE_DEFAULT);
    //fz_features *features = fz_new_page_features(ctx, nullptr);
    fz_drop_page_features(ctx, nullptr);
    
    return strlen(t);
}

%}

#include <stdio.h>
int features_test(const char* t);
