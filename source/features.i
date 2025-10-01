%{
/* 2025-10-01:
    On MacOS we are nnable to do `#include "mupdf/exceptions.h"` because c++ complains about
    MuPDF's FzErrorBase exception class's destructor:

        error: exception specification of overriding function is more lax than base version
           19 | struct FzErrorBase : std::exception
              |        ^
        /Applications/Xcode-15.1.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/c++/v1/__exception/exception.h:78:11: note: overridden virtual function is here
           78 |   virtual ~exception() _NOEXCEPT;
        |           ^

    So for now we don't attempt use `mupdf::internal_throw_exception(ctx);`.
*/

#include "mupdf/classes.h"

#ifdef __APPLE__
#else
    #include "mupdf/exceptions.h" /* See above. */
#endif

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
    
    fz_features* features = nullptr;
    fz_feature_stats feature_stats_ret;
    
    fz_var(features);
    
    fz_try(ctx)
    {
        features = fz_new_page_features(ctx, stext_page.m_internal);
        fz_feature_stats* feature_stats = fz_features_for_region(ctx, features, *region.internal(), category);
        memcpy(&feature_stats_ret, feature_stats, sizeof(feature_stats_ret));
    }
    fz_always(ctx)
    {
        fz_drop_page_features(ctx, features);
    }
    fz_catch(ctx)
    {
        #ifdef __APPLE__
            /* See note above. */
            throw std::runtime_error("fz_features_for_region() failed");
        #else
            mupdf::internal_throw_exception(ctx);
        #endif
    }
    return feature_stats_ret;
}

%}

%include "features_decls.h"

int features_test(const char* t);

/* Python-friendly wrapper for fz_features_for_region(). */
fz_feature_stats fz_features_for_region(mupdf::FzStextPage& stext_page, mupdf::FzRect region, int category);
