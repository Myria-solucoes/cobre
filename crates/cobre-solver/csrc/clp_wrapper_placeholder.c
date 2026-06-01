/* Placeholder link-probe translation unit for the vendored CLP static build.
 *
 * This is a temporary link probe; the real C wrapper (csrc/clp_wrapper.c)
 * replaces it once the CLP FFI layer lands. This file exists only
 * to force the linker to resolve a symbol from the static CLP library, proving
 * that the cmake superbuild produced linkable archives (libClp.a +
 * libCoinUtils.a). It is compiled only when the `clp` feature is enabled.
 *
 * Clp_VersionMajor() is part of the public CLP C interface
 * (Clp_C_Interface.h). We declare it locally rather than including the header
 * so the placeholder needs no include search path of its own.
 */

extern int Clp_VersionMajor(void);

int cobre_clp_placeholder_version_major(void) { return Clp_VersionMajor(); }
