import os
import sys
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

def pngs_to_pdf(png_dir='.', output_pdf='combined_plots.pdf', sort_by_name=True):
    """
    Combine all PNG images in the specified directory into a single PDF file
    
    Parameters:
        png_dir: Directory containing PNG images, default is current directory
        output_pdf: Output PDF filename, default is 'combined_plots.pdf'
        sort_by_name: Whether to sort files by name, default is True
    """
    try:
        # Get all PNG files in the directory
        png_files = [f for f in os.listdir(png_dir) if f.lower().endswith('.png')]
        
        if not png_files:
            print(f"Error: No PNG files found in directory {png_dir}")
            return
        
        # Sort files by name (optional)
        if sort_by_name:
            png_files.sort()
        
        print(f"Found {len(png_files)} PNG files, merging into PDF...")
        print("File list:")
        for i, f in enumerate(png_files, 1):
            print(f"  {i}. {f}")
        
        # Create PDF canvas
        c = canvas.Canvas(output_pdf, pagesize=letter)
        
        for png_file in png_files:
            # Build full path
            full_path = os.path.join(png_dir, png_file)
            
            try:
                # Open image to get dimensions
                with Image.open(full_path) as img:
                    img_width, img_height = img.size
                
                # Calculate scaling ratio to fit PDF page (maintain aspect ratio)
                pdf_width, pdf_height = letter
                scale = min(pdf_width / img_width, pdf_height / img_height) * 0.95  # 5% margin
                
                # Calculate position (centered)
                x = (pdf_width - img_width * scale) / 2
                y = (pdf_height - img_height * scale) / 2
                
                # Add new page and insert image
                c.drawImage(
                    ImageReader(full_path),
                    x, y,
                    width=img_width * scale,
                    height=img_height * scale
                )
                
                # Add filename below the image
                c.setFont("Helvetica", 10)
                c.drawString(x, y - 15, png_file)
                
                # Move to next page (correct method)
                c.showPage()
                
            except Exception as e:
                print(f"Error processing file {png_file}: {str(e)}, skipping this file")
        
        # Save PDF
        c.save()
        print(f"PDF file generated successfully: {os.path.abspath(output_pdf)}")
        
    except Exception as e:
        print(f"Error during merging process: {str(e)}")

if __name__ == "__main__":
    # Support command line arguments: python png_to_pdf.py [image_directory] [output_pdf_name]
    png_dir = '.'
    output_pdf = 'combined_plots.pdf'
    
    if len(sys.argv) > 1:
        png_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_pdf = sys.argv[2]
    
    pngs_to_pdf(png_dir, output_pdf)