(add-to-list 'auto-mode-alist '("\\.adoc\\'" . adoc-mode))

(defvar adoc-mode-hook nil
  "Hook run when entering adoc mode.")

(define-derived-mode adoc-mode text-mode "AsciiDoc"
  "Major mode for editing AsciiDoc files.")

(provide 'adoc-mode)
