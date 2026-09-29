"""Independent PDF rendering, vector/font audits and accessible-color proofs."""
from pathlib import Path
import argparse, hashlib, json, re, shutil, subprocess
import numpy as np
import pandas as pd
import pymupdf
import style
from PIL import Image, ImageOps, ImageDraw, ImageFont

BASE=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-root',type=Path,default=BASE);args=ap.parse_args();out=args.output_root
    qa=out/'qa';proof=qa/'proofs';proof.mkdir(parents=True,exist_ok=True);rows=[];images=[];gray=[];cvd=[]
    env=json.loads((qa/'BUILD_ENVIRONMENT.json').read_text(encoding='utf-8')); expected_font=env['font']
    for i in range(1,5):
        pdf=out/'figures'/f'Figure{i}_final.pdf';doc=pymupdf.open(pdf);p=doc[0]
        if shutil.which('pdftoppm'):
            prefix=proof/f'Figure{i}_PDF_144dpi'
            subprocess.run(['pdftoppm','-r','144','-png','-singlefile',str(pdf),str(prefix)],check=True,capture_output=True)
            renderer='Poppler pdftoppm, 144 dpi'
        else:
            p.get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False).save(proof/f'Figure{i}_PDF_144dpi.png');renderer='MuPDF, 144 dpi'
        image=Image.open(proof/f'Figure{i}_PDF_144dpi.png').convert('RGB');images.append(image.copy())
        im=image.copy();im.thumbnail((1000,1100));im.save(proof/f'Figure{i}_overview.png')
        im=ImageOps.grayscale(image).convert('RGB');im.save(proof/f'Figure{i}_grayscale.png');gray.append(im)
        # Severe deuteranopia preview: linear-RGB projection, clipped only to display gamut.
        rgb=np.asarray(image,dtype=float)/255
        lin=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
        matrix=np.array([[.367322,.860646,-.227968],[.280085,.672501,.047413],[-.011820,.042940,.968881]])
        mapped=np.clip(lin@matrix.T,0,1);srgb=np.where(mapped<=.0031308,mapped*12.92,1.055*mapped**(1/2.4)-.055)
        im=Image.fromarray(np.uint8(np.round(srgb*255)));im.save(proof/f'Figure{i}_deuteranopia.png');cvd.append(im)
        for label,rect in [('upper',pymupdf.Rect(0,0,p.rect.width,p.rect.height*.48)),('lower',pymupdf.Rect(0,p.rect.height*.48,p.rect.width,p.rect.height))]:
            p.get_pixmap(matrix=pymupdf.Matrix(3,3),clip=rect,alpha=False).save(proof/f'Figure{i}_{label}_216dpi.png')
        fonts=p.get_fonts(full=True);embedded=all(len(doc.extract_font(f[0])[3])>0 for f in fonts)
        fonts_names=','.join(f[3] for f in fonts);spans=[s for b in p.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']]
        outside=[s['text'] for s in spans if not p.rect.contains(pymupdf.Rect(s['bbox']))]
        strokes=[x['width'] for x in p.get_drawings() if x.get('color') is not None and x.get('width',0)>0]
        svg=(out/'figures'/f'Figure{i}_final.svg').read_text('utf8')
        font_expected='Arial' if i==1 else expected_font
        normalized_expected=font_expected.replace(' ','').lower()
        fonts_expected=all(normalized_expected in f[3].replace(' ','').lower() or
                           (expected_font=='DejaVu Sans' and 'cmsy10' in f[3].replace(' ','').lower())
                           for f in fonts)
        row=dict(figure=i,pages=len(doc),width_mm=p.rect.width*25.4/72,height_mm=p.rect.height*25.4/72,
                 image_objects=len(p.get_images()),fonts=fonts_names,all_fonts_embedded=embedded,
                 expected_font=font_expected,all_fonts_expected=fonts_expected,svg_image_elements=len(re.findall(r'<image\b',svg)),
                 svg_text_elements=len(re.findall(r'<text\b',svg)),selectable_text_chars=len(p.get_text()),
                 outside_page_text=json.dumps(outside),minimum_stroke_pt=min(strokes) if strokes else None,
                 renderer=renderer,status='PASS' if len(doc)==1 and abs(p.rect.width*25.4/72-180)<.001 and not p.get_images() and embedded and not outside and fonts_expected else 'FAIL')
        rows.append(row)
    def sheet(ims,path):
        w,h=950,1050;canvas=Image.new('RGB',(w*2,h*2),'white');draw=ImageDraw.Draw(canvas)
        try:font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',22)
        except OSError:font=ImageFont.load_default()
        for k,im in enumerate(ims):
            cell=im.copy();cell.thumbnail((w-40,h-65));x=(k%2)*w+20;y=(k//2)*h+45
            draw.text(((k%2)*w+20,(k//2)*h+12),f'Figure {k+1}',font=font,fill='#3F4548');canvas.paste(cell,(x,y))
        canvas.save(path)
    sheet(images,qa/'FIGURE_SET_CONTACT_SHEET.png');sheet(gray,qa/'FIGURE_SET_GRAYSCALE.png');sheet(cvd,qa/'FIGURE_SET_DEUTERANOPIA.png')
    pd.DataFrame(rows).to_csv(qa/'VECTOR_FONT_PAGE_CHECK.csv',index=False)
    # Confirm the packaged inputs remained unchanged during figure rendering.
    immut_path=qa/'BUILD_INPUT_IMMUTABILITY_CHECK.csv'
    immut_frame=pd.read_csv(immut_path) if immut_path.exists() else pd.DataFrame([{'status':'FAIL'}])
    immut_frame.to_csv(qa/'INPUT_IMMUTABILITY_CHECK.csv',index=False)
    immut=immut_frame.to_dict('records')
    # Public math glyphs may use Matplotlib's bundled Computer Modern symbol face.
    print(pd.DataFrame(rows)[['figure','width_mm','height_mm','image_objects','all_fonts_embedded','all_fonts_expected','status']].to_string(index=False))
    assert all(r['status']=='PASS' for r in rows),rows
    assert all(r['status']=='PASS' for r in immut),immut
if __name__=='__main__':main()
