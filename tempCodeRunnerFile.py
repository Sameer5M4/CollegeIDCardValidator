       print("\n--- Raw OCR Output ---")
        try:
            with open(image_to_validate_path, 'rb') as f_img: img_bytes = f_img.read()
            raw_ocr, _ = extract_text_from_image_bytes(img_bytes)
            if raw_ocr:
                for i, line_txt in enumerate(raw_ocr): print(f"  OCR Line {i}: '{line_txt}'")
            else: print("  No text extracted by OCR.")
        except Exception as e_ocr: print(f"  Error OCR debug: {e_ocr}")
        print("--- End of Raw OCR Output ---\n")