import React, { useState } from 'react';
import Form from 'react-bootstrap/Form';
import Papa from 'papaparse';

import { CSVFileType } from '../types';

interface CSVFileInputProps {
  setValue: React.Dispatch<React.SetStateAction<CSVFileType[]>>;
  label: string;
}

const allowedExtensions = ['csv', 'xlsx', 'xls'];

const handleParseFile = (data: string) => {
  const csv = Papa.parse(data, {
    header: true,
    skipEmptyLines: true,
  });
  const parsedData = csv.data as CSVFileType[];
  return parsedData;
};

const CSVFileInput = ({ setValue, label }: CSVFileInputProps) => {
  const [error, setError] = useState('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError('');
    setValue([]);
    // Check if user has entered the file
    if (!e.target?.files?.length) return;

    const inputFile = e.target.files[0];

    // Check the file extensions, if it not
    // included in the allowed extensions
    // we show the error
    const fileExtension = inputFile?.type.split('/')[1];
    if (!allowedExtensions.includes(fileExtension)) {
      setError('Please input a csv file');
      return;
    }

    // Initialize a reader which allows user
    // to read any file or blob.
    const reader = new FileReader();

    // Event listener on reader when the file
    // loads, we parse it and set the data.
    reader.onload = async ({ target }) => {
      if (!target?.result) return;
      const data = handleParseFile(target.result as string);
      setValue(data);
    };
    reader.readAsText(inputFile);
  };

  return (
    <Form.Group controlId='formFile'>
      <Form.Label>
        {label}
        <span className='text-danger'>*</span>
      </Form.Label>
      <input
        className='form-control'
        required
        type='file'
        onChange={handleFileChange}
        accept='.xlsx, .xls, .csv'
      />
    </Form.Group>
  );
};

export default CSVFileInput;
