#!/usr/bin/env Rscript

# Eyeris resource files diagnostic script
# Tests if eyeris package has all required resource files

cat("=== Eyeris Resource Files Diagnostic ===\n\n")

# Test 1: Load eyeris package
cat("1. Loading eyeris package...\n")
tryCatch({
    library(eyeris)
    cat("   SUCCESS: eyeris loaded\n\n")
}, error = function(e) {
    cat(sprintf("   ERROR: %s\n\n", e$message))
    quit(status = 1)
})

# Test 2: Find eyeris package installation directory
cat("2. Finding eyeris package location...\n")
eyeris_path <- system.file(package = "eyeris")
cat(sprintf("   Eyeris path: %s\n", eyeris_path))
cat(sprintf("   Exists: %s\n\n", dir.exists(eyeris_path)))

# Test 3: Check for resource directories
cat("3. Checking for resource directories...\n")
resource_dirs <- c("www", "rmarkdown", "templates", "resources", "inst/www",
                   "inst/rmarkdown", "inst/templates", "inst/resources")

for (rdir in resource_dirs) {
    full_path <- file.path(eyeris_path, rdir)
    exists <- dir.exists(full_path)
    cat(sprintf("   %s: %s\n", rdir, ifelse(exists, "FOUND", "NOT FOUND")))
    if (exists) {
        files <- list.files(full_path, recursive = FALSE)
        cat(sprintf("      Contents: %s\n", paste(head(files, 5), collapse=", ")))
    }
}

# Test 4: Use system.file to find specific resources
cat("\n4. Testing system.file() for common resources...\n")
resources_to_check <- c(
    "www",
    "www/css",
    "www/js",
    "www/images",
    "www/fonts",
    "rmarkdown/templates",
    "resources"
)

for (res in resources_to_check) {
    res_path <- system.file(res, package = "eyeris")
    exists <- nchar(res_path) > 0 && (file.exists(res_path) || dir.exists(res_path))
    cat(sprintf("   %s: %s\n", res, ifelse(exists, res_path, "NOT FOUND")))
}

# Test 5: List all files in eyeris package
cat("\n5. Complete directory structure of eyeris package:\n")
all_files <- list.files(eyeris_path, recursive = TRUE, full.names = FALSE)
cat(sprintf("   Total files: %d\n", length(all_files)))
cat("   First 20 files:\n")
for (f in head(all_files, 20)) {
    cat(sprintf("      %s\n", f))
}

# Test 6: Check eyeris namespace for bidsify function
cat("\n6. Checking eyeris::bidsify function...\n")
if (exists("bidsify", where = "package:eyeris", mode = "function")) {
    cat("   bidsify function: FOUND\n")

    # Try to get the function source to look for resource path references
    cat("   Checking function for resource path references...\n")
    tryCatch({
        func_body <- capture.output(eyeris::bidsify)
        resource_lines <- grep("resource|www|css|js|file.copy|system.file",
                               func_body, value = TRUE, ignore.case = TRUE)
        if (length(resource_lines) > 0) {
            cat("   Found resource-related code:\n")
            for (line in head(resource_lines, 10)) {
                cat(sprintf("      %s\n", trimws(line)))
            }
        } else {
            cat("   No obvious resource path references found\n")
        }
    }, error = function(e) {
        cat(sprintf("   Could not inspect function: %s\n", e$message))
    })
} else {
    cat("   bidsify function: NOT FOUND\n")
}

cat("\n=== Diagnostic Complete ===\n")
cat("\nIf resource directories are NOT FOUND, the eyeris package\n")
cat("installation may be incomplete or corrupted.\n")
