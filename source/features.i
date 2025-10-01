%{
#include "mupdf/classes.h"
#include "mupdf/internal.h"

#include "features_decls.h"

int features_test(const char* t)
{
    fz_context* ctx = fz_new_context(nullptr, nullptr, FZ_STORE_DEFAULT);
    //fz_features *features = fz_new_page_features(ctx, nullptr);
    fz_drop_page_features(ctx, nullptr);
    
    return strlen(t);
}

fz_feature_stats fz_features_for_region(mupdf::FzStextPage& stext_page, mupdf::FzRect region, int category)
{
    fz_context* ctx = mupdf::internal_context_get();
    fz_features* features = fz_new_page_features(ctx, stext_page.m_internal);
    fz_feature_stats* feature_stats = fz_features_for_region(ctx, features, *region.internal(), category);
    fz_feature_stats feature_stats_ret = *feature_stats;
    fz_drop_page_features(ctx, features);
    return feature_stats_ret;
}

%}

%include "features_decls.h"

int features_test(const char* t);

/* Python-friendly wrapper for fz_features_for_region(). */
fz_feature_stats fz_features_for_region(mupdf::FzStextPage& stext_page, mupdf::FzRect region, int category);
