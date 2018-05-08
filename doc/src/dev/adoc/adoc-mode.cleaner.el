(defgroup asciidoc-faces nil
  "AsciiDoc highlighting"
  :group 'asciidoc)

;;;; Title/Section faces

;; Note that Asciidoc supports a document title plus four section
;; levels plus several paragraph-titles ("admonitionn block"). The
;; nomenclature used here borrows from (La)TeX.

(defface asciidoc-document-title-face	; = one
  `((((class color) (background dark))
     (:foreground "DarkGoldenrod3" :bold t :underline t :inherit variable-pitch))
    (((class color) (background light))
     (:foreground "RoyalBlue3" :bold t :underline t :inherit variable-pitch))
    (t (:weight bold :inherit variable-pitch)))
  "Face for AsciiDoc document titles (level 0)."
  :group 'asciidoc-faces)

(defface asciidoc-chapter-face          ; == two
  `((((class color) (background dark))
     (:foreground "DarkGoldenrod3" :bold t :italic t :underline t :inherit variable-pitch))
    (((class color) (background light))
     (:foreground "RoyalBlue3" :bold t :italic t :underline t :inherit variable-pitch))
    (t (:weight bold :inherit variable-pitch)))
  "Face for AsciiDoc section titles (level 1)."
  :group 'asciidoc-faces)

(defface asciidoc-section-face          ; === three
  `((((class color) (background dark))
     (:foreground "DarkGoldenrod1" :bold t :italic t :underline t :inherit variable-pitch))
    (((class color) (background light))
     (:foreground "RoyalBlue2" :bold t :italic t :underline t :inherit variable-pitch))
    (t (:weight bold :inherit variable-pitch)))
  "Face for AsciiDoc section titles (level 2)."
  :group 'asciidoc-faces)

(defface asciidoc-subsection-face       ; ==== four
  `((((class color) (background dark))
     (:foreground "Gold" :bold t :italic t :underline t :inherit variable-pitch))
    (((class color) (background light))
     (:foreground "RoyalBlue2" :bold t :italic t :underline t :inherit variable-pitch))
    (t (:weight bold)))
  "Face for AsciiDoc section titles (level 3)."
  :group 'asciidoc-faces)

(defface asciidoc-subsubsection-face	; ===== five
  `((((class color) (background dark))
     (:foreground "Yellow1" :italic t :underline t :inherit variable-pitch))
    (((class color) (background light))
     (:foreground "RoyalBlue1" :italic t :underline t :inherit variable-pitch))
    (t (:weight bold)))
  "Face for AsciiDoc section titles (level 4)."
  :group 'asciidoc-faces)

(defface asciidoc-paragraph-face
  `((((class color) (background dark))
     (:foreground "Cornsilk" :italic t))
    (((class color) (background light))
     (:foreground "RoyalBlue1" :italic t))
    (t (:inherit variable-pitch)))
  "Face for AsciiDoc paragraph titles and admonition blocks."
  :group 'asciidoc-faces)

(defface asciidoc-comment-face
  `((((class color) (background dark))
     (:bold t :family "Courier" :foreground "LimeGreen"))
    (((class color) (background light))
     (:bold t :family "Courier" :foreground "Forestgreen")))
  "Face for AsciiDoc markup (monospaced text)."
  :group 'asciidoc-faces)

;;;; Markup faces

(defface asciidoc-upper-face
  `((((class color) (background dark))
     (:foreground "Cornsilk"))
    (((class color) (background light))
     (:foreground "DarkGoldenrod1"))
    (t (:weight bold :inherit variable-pitch)))
  "Face for AsciiDoc paragraphs."
  :group 'asciidoc-faces)

(defface asciidoc-mono-face
  `((((class color) (background dark))
     (:bold t :family "Courier"))
    (((class color) (background light))
     (:foreground "DarkMagenta" :bold t :family "Courier")))
  "Face for AsciiDoc markup (monospaced text)."
  :group 'asciidoc-faces)

(defface asciidoc-list-item-face
  `((((class color) (background dark))
     (:foreground "OrangeRed" :bold t :family "Courier"))
    (((class color) (background light))
     (:foreground "DeepskyBlue" :bold t :family "Courier")))
  "Face for AsciiDoc list items."
  :group 'asciidoc-faces)

(defface asciidoc-emph-face
  `((((class color) (background dark))
     (:foreground "Cornsilk" :slant italic))
    (((class color) (background light))
     (:foreground "Gray15" :slant italic)))
  "Face for AsciiDoc markup (emphasized text)."
  :group 'asciidoc-faces)

(defface asciidoc-bold-face
  `((((class color) (background dark))
     (:foreground "Cornsilk" :bold t :inherit variable-pitch))
    (((class color) (background light))
     (:foreground "Gray15" :bold t)))
  "Face for AsciiDoc markup (bold text)."
  :group 'asciidoc-faces)

;;;; Extra faces

(defface asciidoc-idiosyncratic-face
  `((((class color) (background dark))
     (:foreground "IndianRed1" :family "Courier"))
    (((class color) (background light))
     (:foreground "MidnightBlue" :family "Courier"))
    (t ()))
  "Face for AsciiDoc keywords and gibberish, e.g. <<ref>> and url:names[]."
  :group 'asciidoc-faces)

(defface asciidoc-url-face
  `((((class color) (background dark))
     (:foreground "LightGray" :slant italic))
    ;; (:foreground "IndianRed1" :family "Courier"))
    (((class color) (background light))
     (:foreground "MidnightBlue"))
    ;; (:foreground "MidnightBlue" :family "Courier"))
    (t ()))
  "Face for AsciiDoc URLs."
  :group 'asciidoc-faces)

(defface asciidoc-asciimath-face
  `((((class color) (background dark))
     (:foreground "IndianRed1" :family "Courier"))
    (((class color) (background light))
     (:foreground "MidnightBlue" :family "Courier"))
    (t ()))
  "Face for tunneled Asciimath code."
  :group 'asciidoc-faces)

(defface asciidoc-escape-face
  `((((class color) (background dark))
     (:foreground "IndianRed1" :background "Gray10"))
    (((class color) (background light))
     (:foreground "Yellow" :background "Gray20"))
    (t (:weight bold)))
  "Face for unquoted AsciiDoc text."
  :group 'asciidoc-faces)

;;; -------- Major mode setup
;; ----------------------------------------------------------------------------

(defvar asciidoc-mode-hook nil
  "Normal hook run when entering Doc Text mode.")

(defvar asciidoc-mode-abbrev-table nil
  "Abbrev table in use in Asciidoc-mode buffers.")
(define-abbrev-table 'asciidoc-mode-abbrev-table ())

(require 'rx)                             ; http://www.emacswiki.org/emacs/rx
(defmacro asciidoc-rx-markup (&rest term) ; at word boundary
  `(rx ; (or bol ?\( (1+ white))
       ,@term
       ; (any alnum)
       (minimal-match (zero-or-more (not (any "\r")))) ; alias [^\r]*?
       ,@term))

(defmacro asciidoc-rx-markup-nospc (&rest term) ; at word boundary
  `(rx ,@term
       (+? (not space))
       ,@term))

;; (message (asciidoc-rx-markup-nospc ?:))
;; (message (asciidoc-rx-markup       ?'))

(defconst asciidoc-font-lock-keywords
  (eval-when-compile
    (list
     ;; Create a list in preparation to feed it to `font-lock-defaults'. Once a
     ;; piece of text got its coloring, it won't be changed. So, the order 
     ;; is important: the smallest length keyword goes last.
     ;;
     ;; http://www.emacswiki.org/emacs/RegularExpression

     ;; Comment Lines
     (cons "^\\s-*//.*$"         `'asciidoc-comment-face)

     ;; Section titles
     (cons "^=\\s-+.*"           `'asciidoc-document-title-face)
     (cons "^==\\s-+.*"          `'asciidoc-chapter-face)
     (cons "^===\\s-+.*"         `'asciidoc-section-face)
     (cons "^====\\s-+.*"        `'asciidoc-subsection-face)
     (cons "^=====\\s-+.*"       `'asciidoc-subsubsection-face)
     (cons "^======\\s-+.*"      `'asciidoc-paragraph-face)
     (cons "^\\.[A-Za-z0-9ÄÖÜüöäß].*$"`'asciidoc-paragraph-face)

     ;; Asciimath
     (cons "\\$\\$`.+?`\\$\\$"   `'asciidoc-asciimath-face)
     (cons "\\$\\$\\$"           `'asciidoc-asciimath-face)
     (cons "asciimath:\\[.*?\\]" `'asciidoc-asciimath-face)

     ;; URLs
     (cons "\\(?:http\\|ftp\\|email\\)s?://.+?\\[" `'asciidoc-url-face)
     (cons "\\(?:wpde\\|wpen\\):.+?\\[" `'asciidoc-url-face)

     ;; Anchor and cross-references.
     (cons "\\(?:\\[\\[\\[\\sw+,?\\|\\]\\]\\]\\)" `'asciidoc-idiosyncratic-face) ; Bibliographie
     (cons "\\(?:\\[\\[\\sw+,?\\|\\]\\]\\)" `'asciidoc-idiosyncratic-face) ; Anchor
     (cons "\\(?:<<\\sw+\\|>>\\)" `'asciidoc-idiosyncratic-face) ; Reference
     (cons "\\(?:link\\|xref\\|anchor\\):?\\S-+\\[" `'asciidoc-idiosyncratic-face)
     (cons "\\]" `'asciidoc-idiosyncratic-face)

     ;; Images (inline and block)
     (cons "\\(?:image\\)::?\\S-+\\[" `'asciidoc-idiosyncratic-face)

     ;; Preprocessor commands
     (cons "^\\(include\\|sys\\|eval\\|ifn?def\\|endif\\|template\\)[12]?::.*?\\[" `'asciidoc-idiosyncratic-face)

     ;; Footnote
     (cons "footnote:\\[" `'asciidoc-idiosyncratic-face)

     ;; List item paragraphs with implicit numbering
     (cons "^\\s-*\\.\\{1,5\\}\\s-+"	`'asciidoc-list-item-face)

     ;; Bulleted listed item paragraphs
     (cons "^\\s-*-\\s-+"		`'asciidoc-list-item-face)
     (cons "^\\s-*\\*\\{1,5\\}\\s-+"	`'asciidoc-list-item-face)

     ;; List item paragraphs with explicit numbering
     (cons "^\\s-*[0-9]+\\.\\s-+"	`'asciidoc-list-item-face)
     (cons "^\\s-*[a-zA-ZÄÖÜüöäß]\\.\\s-+" `'asciidoc-list-item-face)
     (cons "^\\s-*[ixcvmIXCVM]+)\\s-+"	`'asciidoc-list-item-face)

     ;; Labeled list items
     (cons "^.*[:;][:;-]\\s-" `'asciidoc-list-item-face)

     ;; Special lines that continue list items
     (cons "^\\(?:\\+\\|--\\)\\s-*$" `'asciidoc-idiosyncratic-face)

     ;; Delimited blocks
     (cons "^[_=\\.\\*\\+\\-]\\{6,\\}\\s-*$" `'asciidoc-idiosyncratic-face)
     (cons (asciidoc-rx-markup-nospc ?:) `'asciidoc-idiosyncratic-face)

     ;; Admonition blocks and annotations.
     (cons (concat "\\<\\(?:TODO\\|BUG\\|ERROR\\|DISCLAIMER\\|WARNING\\|NOTE"
                   "\\|ERROR\\|TIP\\|CAUTION\\|IMPORTANT\\|EXAMPLE\\|BEISPIEL\\):") `'asciidoc-paragraph-face)
     (cons "\\*[A-ZÄÖÜüöäß]+\\*:" `'asciidoc-paragraph-face)

     ;; Super/subscript
     (cons (asciidoc-rx-markup ?^) `'asciidoc-emph-face)
     (cons (asciidoc-rx-markup ?~) `'asciidoc-emph-face)

     ;; Uppercase
     (cons "\\b[A-ZÄÖÜß]+[A-ZÜÖÄß0-9\\- ]*\\b" `'asciidoc-upper-face)

     ;; Embedded HTML entities.
     (cons "&#[xX]?[0-9a-fA-F]+?;" `'font-lock-constant-face)
     (cons "#[xX]?[0-9a-fA-F]+?;" `'font-lock-constant-face)

     ;; Literal text.
     (cons "\\(?:(R)\\|(TM)\\|(C)\\|---\\|--\\|\\.\\.\\.\\|[=-]+[<>]\\)" `'font-lock-builtin-face)

     ;; Unquoted text and trailing whitespace
     (cons "\\(?:##[^\r]*?##\\)" `'asciidoc-escape-face)
     (cons "\\(?:#.*?#\\)" `'asciidoc-escape-face)
     (cons "[ \t\v\r]+$" `'asciidoc-escape-face)

     ;; Normal markups (word boundaries) and unconstrained markups. Mono, bold,
     ;; emphasized. Quite a challenge, since the fontlock feature was build to
     ;; highlight single lines only, but markups can go over multiple lines.

     (cons (asciidoc-rx-markup ?+) `'asciidoc-mono-face)
     (cons (asciidoc-rx-markup ?*) `'asciidoc-bold-face)
     (cons (asciidoc-rx-markup ?') `'asciidoc-emph-face)

     ;; Literal text (quoted text) that can span multiple lines.
     ;;
     ;; To match newlines in Emacs use the negative clause "[^\r]+", which
     ;; matches anything that is not specified, including newlines. "\r" is just
     ;; one rarely used character under UN*X. To match single lines use "\s-+"
     ;; or "[^ \t]+" which matches non-whitespace characters.
     ;;
     ;; However, "[^\r]" highlights multiple lines rather accidently. When you
     ;; edit lines the highlight possibly disappears. The solution is to run
     ;; `normal-mode' again to reset fontlocking (maybe timer-triggered).

     (cons "\"[^\r]+?\"" `'font-lock-string-face)
     (cons "``[^\r]+?''" `'font-lock-string-face)
     (cons "`[^\r]+?'" `'font-lock-string-face)
     )
    )
  "Syntax expressions in AsciiDoc editing mode.")

;;###autoload
(define-derived-mode asciidoc-mode text-mode "AsciiDoc"
  "Major mode for outlined AsciiDoc text files.

Since this mode, derives from `text-mode', and enables `outline-minor-mode', it
runs, in that order, `text-mode-hook', `outline-minor-mode-hook' and finally
`asciidoc-mode-hook'."
  (interactive)
  (message "asciidoc-mode")
  (turn-on-auto-fill)
  (not-modified)

  (modify-syntax-entry ?\' ".")
  (make-local-variable 'paragraph-start)
  (make-local-variable 'paragraph-separate)
  (make-local-variable 'paragraph-ignore-fill-prefix)
  (make-local-variable 'require-final-newline)
  (make-local-variable 'font-lock-defaults)

  (setq comment-start "// "
        paragraph-start (concat "$\\|>" page-delimiter)
        paragraph-separate paragraph-start
        paragraph-ignore-fill-prefix t
        require-final-newline t
        case-fold-search t
        font-lock-defaults '(asciidoc-font-lock-keywords nil nil ((?_ . "w"))))

  ;; Insert, align, or delete end-of-line backslashes on the lines in the
  ;; region. See also `c-backslash-region'.
  (local-set-key [?\C-c ?\\] 'makefile-backslash-region)

  ;; Enable minor outline mode.
  (message "loading outline")
  (require 'outline)
  (message "enabling outline-minor-mode")
  (outline-minor-mode)
  (set (make-local-variable 'outline-regexp) "^[=]+ ")

  ;; Run our own mode hook in the sequence.
  (message "running asciidoc-mode-hook")
  (run-mode-hooks 'asciidoc-mode-hook)
  (message "%s: asciidoc-mode" (buffer-name (current-buffer)))

  ;; Define code skeletons.


  )

(eval-after-load "asciidoc-mode"
  '(progn
     (define-skeleton sk-quote "AsciiDoc quote paragraph." nil
       \n "[quote,,]" \n
       "____________________" \n - \n
       "____________________" \n \n)
     ;; (define-abbrev cperl-mode-abbrev-table "yquote" "" 'sk-quote)
     )
  )

(provide 'asciidoc-mode)

;;; asciidoc-mode.el ends here
