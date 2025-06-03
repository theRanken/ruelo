def base64_string_looks_like_image(s):
        """
        Helper function to determine if a long string is likely a base64 encoded image.
        This is a heuristic and not foolproof.
        """
        # Check for common base64 characters and length
        return len(s) > 100 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in s[-100:])