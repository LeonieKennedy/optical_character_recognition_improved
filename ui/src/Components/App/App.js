import './App.css';
import React, { useEffect, useReducer, useState } from 'react';
import Container from '@material-ui/core/Container'
import { Button, FormControlLabel, Switch, Tooltip, Typography } from '@material-ui/core'
import Checkbox from "@material-ui/core/Checkbox";
import TextField from '@mui/material/TextField';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';
import { Autocomplete, LoadingButton } from '@mui/lab';
import { Stack } from '@mui/system';

const formReducer = (state, event) => {
  if(event.reset) {
    return {
      selected_model: '',
      file: '',
      text: '',
      // Tesseract
      scale: 1,
      // EasyOCR
      language: '',
      paragraph: false,
      custom_buffer: 0,
      //  Advanced options
      advanced: false,
      thresholding: false,
      skew_correction: false,
      noise_removal: false,
    }
  }
 return {
   ...state,
   [event.name]: event.value
 }
}

function App() {
  const [formData, setFormData] = useReducer(formReducer, {});    // Inputs from the form
  const [ocr_output, ocr_setOutput] = useState(String);           // Output from models
  const [change, setChange] = useState(false);
  const [loading, setLoading] = useState(false);
  const language_choices = [
    'Abaza',
    'Adyghe',
    'Afrikaans',
    'Angika',
    'Arabic',
    'Assamese',
    'Avar',
    'Azerbaijani',
    'Belarusian',
    'Bulgarian',
    'Bihari',
    'Bhojpuri',
    'Bengali',
    'Bosnian',
    'Simplified Chinese',
    'Traditional Chinese',
    'Chechen',
    'Czech',
    'Welsh',
    'Danish',
    'Dargwa',
    'German',
    'English',
    'Spanish',
    'Estonian',
    'Persian (Farsi)',
    'French',
    'Irish',
    'Goan Konkani',
    'Hindi',
    'Croatian',
    'Hungarian',
    'Indonesian',
    'Ingush',
    'Icelandic',
    'Italian',
    'Japanese',
    'Kabardian',
    'Kannada',
    'Korean',
    'Kurdish',
    'Latin',
    'Lak',
    'Lezghian',
    'Lithuanian',
    'Latvian',
    'Magahi',
    'Maithili',
    'Maori',
    'Mongolian',
    'Marathi',
    'Malay',
    'Maltese',
    'Nepali',
    'Newari',
    'Dutch',
    'Norwegian',
    'Occitan',
    'Pali',
    'Polish',
    'Portuguese',
    'Romanian',
    'Russian',
    'Serbian (cyrillic)',
    'Serbian (latin)',
    'Nagpuri',
    'Slovak',
    'Slovenian',
    'Albanian',
    'Swedish',
    'Swahili',
    'Tamil',
    'Tabassaran',
    'Telugu',
    'Thai',
    'Tajik',
    'Tagalog',
    'Turkish',
    'Uyghur',
    'Ukranian',
    'Urdu',
    'Uzbek',
    'Vietnamese'
      ]                                                           // EasyOCR language mapping

  const handleSubmit = event => {
    event.preventDefault();
    setChange(false);
  }

  const handleChange = event => {
    setChange(false);
    setLoading(false);
    const isCheckbox = event.target.type === 'checkbox';
    setFormData({
      name: event.target.name,
      value: isCheckbox ? event.target.checked : event.target.value,
    });
  };

  useEffect(() => {
    console.log("selected model: " + formData.selected_model)
    if(formData.selected_model === 'tesseract') {
      tesseract();
    }
    if(formData.selected_model=== 'easyocr') {
      easyocr();
    }
    if (formData.selected_model === 'keras') {
      keras();
    }
  }, []);

  const tesseract = async () => {
    setLoading(true);
    
    const fileData = new FormData();
    fileData.append("image_file_path", formData.file);
    fileData.append("type", formData.file.type);

    const response = await fetch(
      'http://localhost:8000/tesseract?&scale=' + (formData.scale || 1) + "&thresholding=" + (formData.thresholding || false) + "&skew_correction=" + (formData.skew_correction || false) + "&noise_removal=" + (formData.noise_removal || false),
    {
      method: "POST",
      body: fileData,
    })
    .then((response) => {
      return response.json();
    })
    .then(data => {
      console.log(data)
      ocr_setOutput(Object.values(data["text"]));
      setChange(true);
    })
    setLoading(false);
  }

  const easyocr = async () => {
    setLoading(true);
    
    const fileData = new FormData();
    fileData.append("image_file_path", formData.file);
    fileData.append("type", formData.file.type);

    const response = await fetch(
      'http://localhost:8000/easyocr?language=' + (formData.language || "English") + '&paragraph=' + (formData.paragraph || false) + "&thresholding=" + (formData.thresholding || false) + "&skew_correction=" + (formData.skew_correction || false) + "&noise_removal=" + (formData.noise_removal || false),
    {
      method: "POST",
      body: fileData,
    })
    .then((response) => {
      return response.json();
    })
    .then(data => {
      console.log(data)
      ocr_setOutput(Object.values(data["text"]));
      setChange(true);
    })
    setLoading(false);

  }

  const keras = async () => {
    setLoading(true);
    const fileData = new FormData();
    fileData.append("image_file_path", formData.file);
    fileData.append("type", formData.file.type);
    
    const response = await fetch(
      'http://localhost:8000/keras?thresholding=' + (formData.thresholding || false) + "&skew_correction=" + (formData.skew_correction || false) + "&noise_removal=" + (formData.noise_removal || false),
    {
      method: "POST",
      body: fileData,
    })
    .then((response) => {
      return response.json();
    })
    .then(data => {
      ocr_setOutput(Object.values(data["text"]));
      setChange(true);
    })
    setLoading(false);
  }

  return (
    <Container maxWidth="md" className="App">
      <h1>OCR</h1>
      <form onSubmit={handleSubmit}>

{/* --------------------------- Dropdown Options --------------------------------- */}
        <Stack spacing={2} sx={{width: "30%", marginLeft: '35%', marginRight: '35%'}}>
          <InputLabel required>Choose a model:</InputLabel>
          <Select
            name="selected_model"
            id="dropdown"
            label="Option"
            value={formData.selected_model || ''}
            onChange={handleChange}
          >
            <MenuItem value={"tesseract"}>Tesseract</MenuItem>
            <MenuItem value={"keras"}>Keras</MenuItem>
            <MenuItem value={"easyocr"}>EasyOCR</MenuItem>
          </Select>

    {/* --------------------------- Get image file ----------------------------------- */}
          <Button variant='contained' color="primary" component='label'>
            <input 
              accept="image/png/*" 
              type='file'
              onChange={event => {
              setLoading(false);
              setFormData({
              name: 'file',
              value: event.target.files[0]})} }
              required
            />
          </Button>

    {/* --------------------------- Advanced options? -------------------------------- */}
          <Tooltip title="Pre-processing options">
            <FormControlLabel control={
              <Switch 
                checked={formData.advanced || false} 
                onChange={handleChange} 
                name="advanced"
              />}
              label="advanced" 
            />
          </Tooltip>
        </Stack>
        <br></br>

{/* ---------------------------- Advanced Options -------------------------------- */}
        { formData.advanced === true && 
          <fieldset>
            <legend>Advanced Options: Only select 1</legend>
            <Tooltip title="Convert image to binary">
              <FormControlLabel control={
                <Switch checked={formData.thresholding} onChange={handleChange} name="thresholding"/>} 
                  label="Adaptive Thresholding" />
            </Tooltip>

            <Tooltip title={
              <React.Fragment>
                <Typography color="inherit">Correct image rotation</Typography>
                {"Can be slow and can sometimes skew images which are already oriented correctly"}
              </React.Fragment>
              } followCursor>
              <FormControlLabel control={
                <Switch checked={formData.skew_correction} onChange={handleChange} name="skew_correction"/>} 
                  label="Skew Correction" />  
            </Tooltip>

            <Tooltip title="Remove noise">
              <FormControlLabel control={
                <Switch checked={formData.noise_removal} onChange={handleChange} name="noise_removal"/>} 
                  label="Noise Removal" />     
            </Tooltip>

          </fieldset>
        }
  
{/* ------------------------- Tesseract Options ---------------------------------- */}  
        { formData.selected_model === 'tesseract' &&
          <fieldset>
            <legend>Tesseract Options</legend>    
            <Stack spacing={2} direction="column" justifyContent="center" alignItems="center">        
              <TextField        
                  id="outlined-name"
                  name="scale"
                  label="Scale: default k=1"
                  value={formData.scale || ""}
                  onChange={handleChange}
                />
                <LoadingButton 
                  type="submit"
                  loading={loading} 
                  variant="contained" 
                  loadingPosition="center"
                  color="primary" 
                  onClick={tesseract}
                  >
                  Extract Text
                </LoadingButton>
            </Stack>   
          </fieldset>
        }
        <br></br>
    
{/* --------------------------- Keras  Options ----------------------------------- */}
        { formData.selected_model === 'keras' &&
          <fieldset>
            <legend>Keras Options</legend>
            <Stack spacing={2} direction="column" justifyContent="center" alignItems="center"> 
              <LoadingButton 
                type="submit"
                loading={loading} 
                variant="contained" 
                loadingPosition="center"
                color="primary" 
                onClick={keras}
                >
                Extract Text
              </LoadingButton>
            </Stack>
          </fieldset>
        }

{/* -------------------------- EasyOCR Options ----------------------------------- */}
        { formData.selected_model === 'easyocr' &&
          <fieldset>
            <legend>EasyOCR Options</legend>
            <Stack spacing={2}  direction="column" justifyContent="center" alignItems="center">
              <Autocomplete
                sx={{width: "250px"}}
                clearOnEscape
                value={formData.language}             
                options={language_choices}
                onChange={(event) =>     
                  setFormData({
                    name: 'language',
                    value: event.target.innerText,
                  })
                }
                renderInput={(params) => 
                  <TextField {...params}
                    label="Language"
                  />
                }
              />

              <div>Paragraph</div>
              <Checkbox          
                label="paragraph" 
                name="paragraph" 
                checked={formData['paragraph'] || false} 
                onChange={handleChange}
              />

              <LoadingButton 
                type="submit"
                loading={loading} 
                variant="contained" 
                loadingPosition="center"
                color="primary" 
                onClick={easyocr}
                >
                Extract Text
              </LoadingButton>
            </Stack>
          </fieldset>
        }

{/* --------------------------------- Output ------------------------------------ */}
        {change &&
          <fieldset> 
            <legend>Output</legend>     
          { formData.selected_model === 'keras' &&
            <div>
              <img src={require('../images/annotated_keras.jpg')} height="100%" width="100%"/>
            </div>
          }
            <div>
              <strong>Extacted Text: </strong>
              <p className="with-breaks">{ocr_output}</p>
            </div>
          </fieldset>
        }   
    
{/* ------------------------------ Reset Button ---------------------------------- */}
        <p>
          <Button type="reset" 
            variant="contained" 
            color="primary" 
            onClick={() => {
              setChange(false);
              setFormData({reset: true})
              ocr_setOutput([])
              }
            }>
            Reset
          </Button>
        </p>
      </form>
    
      <br></br>
      <br></br>
       
    </Container>
  );
}

export default App;
