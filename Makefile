# GNU Makefile

-include user.make

ifndef build
  build := release
endif

default: all

include Makerules

ifndef OUT
  OUT := build/$(build_prefix)$(build)$(build_suffix)
endif

#include Makethird

# --- Configuration ---

CFLAGS = -Imupdf/include

# Do not specify CFLAGS, LDFLAGS, LIB_LDFLAGS, EXE_LDFLAGS or LIBS on the make
# invocation line - specify XCFLAGS, XLDFLAGS, XLIB_LDFLAGS, XEXE_LDFLAGS or
# XLIBS instead. Make ignores any lines in the makefile that set a variable
# that was set on the command line.
CFLAGS += $(XCFLAGS) -Iinclude
LIBS += $(XLIBS) -lm

LDFLAGS += $(XLDFLAGS)
LIB_LDFLAGS += $(XLIB_LDFLAGS)
EXE_LDFLAGS += $(XEXE_LDFLAGS)

# --- Commands ---

ifneq ($(verbose),yes)
  QUIET_AR = @ echo "    AR $@" ;
  QUIET_RANLIB = @ echo "    RANLIB $@" ;
  QUIET_CC = @ echo "    CC $@" ;
  QUIET_CXX = @ echo "    CXX $@" ;
  QUIET_GEN = @ echo "    GEN $@" ;
  QUIET_LINK = @ echo "    LINK $@" ;
  QUIET_RM = @ echo "    RM $@" ;
  QUIET_TAGS = @ echo "    TAGS $@" ;
  QUIET_WINDRES = @ echo "    WINDRES $@" ;
  QUIET_OBJCOPY = @ echo "    OBJCOPY $@" ;
  QUIET_DLLTOOL = @ echo "    DLLTOOL $@" ;
  QUIET_GENDEF = @ echo "    GENDEF $@" ;
endif

MKTGTDIR = mkdir -p $(dir $@)
CC_CMD = $(QUIET_CC) $(MKTGTDIR) ; $(CC) $(CFLAGS) -MMD -MP -o $@ -c $<
CXX_CMD = $(QUIET_CXX) $(MKTGTDIR) ; $(CXX) $(CFLAGS) $(XCXXFLAGS) -MMD -MP -o $@ -c $<
AR_CMD = $(QUIET_AR) $(MKTGTDIR) ; $(AR) cr $@ $^
ifdef RANLIB
  RANLIB_CMD = $(QUIET_RANLIB) $(RANLIB) $@
endif
LINK_CMD = $(QUIET_LINK) $(MKTGTDIR) ; $(CC) $(LDFLAGS) -o $@ $^ $(LIBS)
TAGS_CMD = $(QUIET_TAGS) ctags
WINDRES_CMD = $(QUIET_WINDRES) $(MKTGTDIR) ; $(WINDRES) $< $@
OBJCOPY_CMD = $(QUIET_OBJCOPY) $(MKTGTDIR) ; $(LD) -r -b binary -z noexecstack -o $@ $<
GENDEF_CMD = $(QUIET_GENDEF) gendef - $< > $@
DLLTOOL_CMD = $(QUIET_DLLTOOL) dlltool -d $< -D $(notdir $(^:%.def=%.dll)) -l $@

ifeq ($(shared),yes)
LINK_CMD = $(QUIET_LINK) $(MKTGTDIR) ; $(CC) $(LDFLAGS) -o $@ \
	$(filter-out %.$(SO)$(SO_VERSION),$^) \
	$(sort $(patsubst %,-L%,$(dir $(filter %.$(SO)$(SO_VERSION),$^)))) \
	$(patsubst lib%.$(SO)$(SO_VERSION),-l%,$(notdir $(filter %.$(SO)$(SO_VERSION),$^))) \
	$(LIBS)
endif

# --- Rules ---

$(OUT)/%.a :
	$(AR_CMD)
	$(RANLIB_CMD)

$(OUT)/%.exe: %.c
	$(LINK_CMD)

$(OUT)/source/%.o : source/%.c
	$(CC_CMD) $(WARNING_CFLAGS) -Wdeclaration-after-statement $(LIB_CFLAGS) $(THIRD_CFLAGS)

.PRECIOUS : $(OUT)/%.o # Keep intermediates from chained rules
.PRECIOUS : $(OUT)/%.exe # Keep intermediates from chained rules

THEIR_LIBMUPDF := build/$(build)/libmupdf.a
THEIR_LIBTHIRDPARTY := build/$(build)/libmupdf-third.a
LIBMUPDF = mupdf/$(THEIR_LIBMUPDF)
LIBTHIRDPARTY = mupdf/$(THEIR_LIBTHIRDPARTY)

mupdf: $(LIBMUPDF) $(LIBTHIRDPARTY)

$(LIBMUPDF):
	cd mupdf && make build=$(build) $(THEIR_LIBMUPDF)
$(LIBTHIRDPARTY):
	cd mupdf && make build=$(build) $(THEIR_THIRDPARTY)

# --- File lists ---

UTILS_SRC := source/csv.c source/utils.c
CANDIDATES_SRC := source/candidates.c
FEATURES_SRC := source/features.c
PICKER_SRC := source/picker.c

UTILS_OBJ := $(UTILS_SRC:%.c=$(OUT)/%.o)
CANDIDATES_OBJ := $(CANDIDATES_SRC:%.c=$(OUT)/%.o)
FEATURES_OBJ := $(FEATURES_SRC:%.c=$(OUT)/%.o)
PICKER_OBJ := $(PICKER_SRC:%.c=$(OUT)/%.o)

CANDIDATES_EXE := $(OUT)/candidates$(EXE)
FEATURES_EXE := $(OUT)/features$(EXE)
PICKER_EXE := $(OUT)/picker$(EXE)

$(CANDIDATES_EXE) : $(CANDIDATES_OBJ) $(UTILS_OBJ) $(LIBMUPDF) $(LIBTHIRDPARTY)
	$(LINK_CMD) $(EXE_LDFLAGS)
$(FEATURES_EXE) : $(FEATURES_OBJ) $(UTILS_OBJ) $(LIBMUPDF) $(LIBTHIRDPARTY)
	$(LINK_CMD) $(EXE_LDFLAGS)
$(PICKER_EXE) : $(PICKER_OBJ) $(UTILS_OBJ) $(LIBMUPDF) $(LIBTHIRDPARTY)
	$(LINK_CMD) $(EXE_LDFLAGS)

all: apps

features: $(FEATURES_EXE)
candidates: $(CANDIDATES_EXE)
picker: $(PICKER_EXE)

apps: features candidates picker

clean:
	rm -rf $(OUT)
nuke:
	rm -rf build/*

release:
	$(MAKE) build=release
debug:
	$(MAKE) build=debug
sanitize:
	$(MAKE) build=sanitize

.PHONY: all clean nuke install third libs apps generate tags docs
.PHONY: shared shared-debug shared-clean
.PHONY: c++-% python-% csharp-%
.PHONY: c++-clean python-clean csharp-clean

