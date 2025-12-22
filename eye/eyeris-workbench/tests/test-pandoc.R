#!/usr/bin/env Rscript

# Pandoc diagnostics script
# Run this to test pandoc configuration without running full pipeline

cat("=== Pandoc Diagnostic Test ===\n\n")

# Test 1: Check if pandoc is in PATH
cat("1. Checking pandoc in PATH...\n")
pandoc_binary <- tryCatch(
    system("which pandoc", intern = TRUE),
    error = function(e) ""
)
cat(sprintf("   Pandoc binary: %s\n", pandoc_binary))
cat(sprintf("   Exists: %s\n\n", file.exists(pandoc_binary)))

# Test 2: Check pandoc version directly
cat("2. Checking pandoc version (direct call)...\n")
tryCatch({
    version_output <- system("pandoc --version", intern = TRUE)
    cat(sprintf("   %s\n\n", version_output[1]))
}, error = function(e) {
    cat(sprintf("   ERROR: %s\n\n", e$message))
})

# Test 3: Check pandoc data directory
cat("3. Checking pandoc data directory...\n")
tryCatch({
    data_dir <- system("pandoc --print-default-data-file templates/default.html 2>&1", intern = TRUE)
    cat(sprintf("   %s\n\n", data_dir))
}, error = function(e) {
    cat(sprintf("   ERROR: %s\n\n", e$message))
})

# Test 4: Check if rmarkdown package is available
cat("4. Checking rmarkdown package...\n")
if (requireNamespace("rmarkdown", quietly = TRUE)) {
    cat("   rmarkdown package: INSTALLED\n")

    # Test 5: Try rmarkdown::find_pandoc()
    cat("\n5. Testing rmarkdown::find_pandoc()...\n")
    tryCatch({
        if (nchar(pandoc_binary) > 0 && file.exists(pandoc_binary)) {
            pandoc_dir <- dirname(pandoc_binary)
            rmarkdown::find_pandoc(dir = pandoc_dir)
            cat(sprintf("   Pandoc directory: %s\n", pandoc_dir))
            cat(sprintf("   Pandoc version (rmarkdown): %s\n", rmarkdown::pandoc_version()))
        } else {
            cat("   Skipping - pandoc binary not found\n")
        }
    }, error = function(e) {
        cat(sprintf("   ERROR: %s\n", e$message))
    })
} else {
    cat("   rmarkdown package: NOT INSTALLED\n")
}

# Test 6: Try to render a minimal Rmd document
cat("\n6. Testing minimal R Markdown render...\n")

# Create a minimal test Rmd file
test_rmd <- tempfile(fileext = ".Rmd")
cat('---
title: "Pandoc Test"
output: html_document
---

# Test Header

This is a test.

```{r}
1 + 1
```
', file = test_rmd)

cat(sprintf("   Test file: %s\n", test_rmd))

tryCatch({
    output_file <- rmarkdown::render(
        test_rmd,
        output_format = "html_document",
        quiet = TRUE
    )
    cat(sprintf("   SUCCESS! Output: %s\n", output_file))
    cat(sprintf("   Output exists: %s\n", file.exists(output_file)))

    # Clean up
    if (file.exists(output_file)) unlink(output_file)
}, error = function(e) {
    cat(sprintf("   ERROR: %s\n", e$message))
    cat("\n   Full error details:\n")
    print(e)
})

# Clean up temp file
if (file.exists(test_rmd)) unlink(test_rmd)

cat("\n=== Diagnostic Test Complete ===\n")
