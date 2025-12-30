from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from pathlib import Path

SLIDES = [
    ("Title", "Acme AI: Automating Data Workflows"),
    ("Problem", "Teams waste hours weekly cleaning and moving data, slowing decisions."),
    ("Solution", "A no-code/LLM-powered automation platform that integrates with modern stacks."),
    ("Market Size", "TAM $12B | SAM $4B | SOM $600M; growing 15% CAGR."),
    ("Product", "Drag-and-drop pipelines, GPT-based transforms, observability, and alerts."),
    ("Business Model", "SaaS: Pro $99/mo, Team $499/mo, Enterprise custom; usage-based overages."),
    ("Traction", "500 logos; $120k MRR; 6% MoM growth; NRR 118%; churn 2.1%."),
    ("Go-to-Market", "Bottom-up PLG + partner channels; content, webinars, and community."),
    ("Competition", "Airflow, Fivetran, dbt Cloud; we win on speed-to-value and LLM-native UX."),
    ("Unit Economics", "CAC $410; LTV $5,200; Payback 6.5 months; Gross margin 82%."),
    ("Roadmap", "Fine-tuned agents, enterprise SSO, governance, and SOC2 Type II."),
    ("Team", "Ex-Google, Ex-Snowflake; 2nd-time founders; 12 engineers, 3 GTM."),
    ("Financials", "ARR $1.44M; Burn $110k/mo; 18 months runway; raising $3M seed."),
    ("Ask", "$3M to accelerate GTM; key hires: Head of Sales, Sr. ML Engineer.")
]

def draw_header_footer(c: canvas.Canvas, page: int, total: int):
    width, height = LETTER
    c.setStrokeColor(colors.HexColor('#7C3AED'))
    c.setFillColor(colors.HexColor('#7C3AED'))
    c.rect(0, height-0.25*inch, width, 0.25*inch, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont('Helvetica-Bold', 12)
    c.drawString(0.5*inch, height-0.18*inch, 'Pitchdeck AI — Sample Deck')
    c.setFillColor(colors.grey)
    c.setFont('Helvetica', 9)
    c.drawString(0.5*inch, 0.3*inch, f'Page {page}/{total}')

def draw_slide(c: canvas.Canvas, title: str, body: str):
    width, height = LETTER
    # Title
    c.setFillColor(colors.HexColor('#111827'))
    c.setFont('Helvetica-Bold', 22)
    c.drawString(0.75*inch, height-1.25*inch, title)
    # Body text
    c.setFont('Helvetica', 12)
    c.setFillColor(colors.HexColor('#111827'))
    x = 0.75*inch
    y = height-1.75*inch
    max_width = width-1.5*inch
    for para in body.split('\n'):
        words = para.split(' ')
        line = ''
        for w in words:
            test = (line + ' ' + w).strip()
            if c.stringWidth(test, 'Helvetica', 12) <= max_width:
                line = test
            else:
                c.drawString(x, y, line)
                y -= 16
                line = w
        if line:
            c.drawString(x, y, line)
            y -= 16
        y -= 8

def create_pdf(path: str):
    # Ensure destination directory exists
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(SLIDES)
    c = canvas.Canvas(str(out_path), pagesize=LETTER)
    for i, (title, body) in enumerate(SLIDES, start=1):
        draw_header_footer(c, i, total)
        draw_slide(c, title, body)
        c.showPage()
    c.save()

if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else 'sample_pitch.pdf'
    create_pdf(out)
    print(f'Wrote {out}')
