from flask import Flask, request, render_template
import deploy as dp
import os as os
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def index():
    temp_img_path = '/Bird Species/static/temp_img.jpg'
    if request.method == 'POST':
        # Get the image file from the HTML form
        img_file = request.files['file-input']
        img_data = img_file.read()  # Read image data as bytes
        
        # Save image data to a temporary file
        
        with open(temp_img_path, 'wb') as temp_img_file:
            temp_img_file.write(img_data)
        
        # Pass the temporary image file path to the output function
        res = dp.output(temp_img_path)    
        return render_template('main.html', img_data=res)
            # Remove the temporary file
    

    return render_template('main.html', img_data=None)

if __name__ == '__main__':
    app.run(debug=True)
