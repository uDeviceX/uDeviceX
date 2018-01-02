(defun u/c-indent-common ()
  (setq c-basic-offset 4
	indent-tabs-mode nil))

(defun u/c++-indent ()
  (u/c-indent-common)
  (c-set-offset 'innamespace [0]))

(defun u/c-indent () (u/c-indent-common))

(add-hook 'c-mode-hook   'u/c-indent)
(add-hook 'c++-mode-hook 'u/c++-indent)

(setq c-default-style
      '((java-mode . "java")
        (awk-mode  . "awk")
        (c-mode    . "k&r")
        (cc-mode   . "k&r")))

(add-to-list 'auto-mode-alist '("\\.cu\\'" . c++-mode))
(add-to-list 'auto-mode-alist '("\\.h\\'"  . c++-mode))
