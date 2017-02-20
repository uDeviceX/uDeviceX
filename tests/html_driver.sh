list=test_data

mkdir -p html

for d in $list; do
    cp "$d"/*.png html
done

./inc_to_html.awk index.templ.html > html/index.html
