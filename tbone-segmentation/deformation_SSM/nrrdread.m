function [X, meta] = nrrdread(filename, segment_id)
%NRRDREAD  Import NRRD imagery and metadata.
%   [X, META] = NRRDREAD(FILENAME) reads the image volume and associated
%   metadata from the NRRD-format file specified by FILENAME.
%
%   Current limitations/caveats:
%   * "Block" datatype is not supported.
%   * Only tested with "gzip" and "raw" file encodings.
%   * Very limited testing on actual files.
%   * I only spent a couple minutes reading the NRRD spec.
%
%   See the format specification online:
%   http://teem.sourceforge.net/nrrd/format.html

% Copyright 2012 The MathWorks, Inc.


% Open file.
fid = fopen(filename, 'rb');
assert(fid > 0, 'Could not open file.');
cleaner = onCleanup(@() fclose(fid));

% Magic line.
theLine = fgetl(fid);
assert(numel(theLine) >= 4, 'Bad signature in file.')
assert(isequal(theLine(1:4), 'NRRD'), 'Bad signature in file.')

% The general format of a NRRD file (with attached header) is:
% 
%     NRRD000X
%     <field>: <desc>
%     <field>: <desc>
%     # <comment>
%     ...
%     <field>: <desc>
%     <key>:=<value>
%     <key>:=<value>
%     <key>:=<value>
%     # <comment>
% 
%     <data><data><data><data><data><data>...

meta = struct([]);

% Parse the file a line at a time.
while (true)

  theLine = fgetl(fid);
  
  if (isempty(theLine) || feof(fid))
    % End of the header.
    break;
  end
  
  if (isequal(theLine(1), '#'))
      % Comment line.
      continue;
  end
  
  % "fieldname:= value" or "fieldname: value" or "fieldname:value"
  parsedLine = regexp(theLine, ':=?\s*', 'split','once');
  
  assert(numel(parsedLine) == 2, 'Parsing error')
  
  field = lower(parsedLine{1});
  value = parsedLine{2};
  
  field(isspace(field)) = '';
  meta(1).(field) = value;
  
end

datatype = getDatatype(meta.type);

% Get the size of the data.
assert(isfield(meta, 'sizes') && ...
       isfield(meta, 'dimension') && ...
       isfield(meta, 'encoding'), ...
       'Missing required metadata fields.')

dims = sscanf(meta.sizes, '%d');
ndims = sscanf(meta.dimension, '%d');
assert(numel(dims) == ndims);

data = readData(fid, meta, datatype);
if isfield(meta,'endian')
    data = adjustEndian(data, meta);
end
% Reshape and get into MATLAB's order.
X = reshape(data, dims');
dims = size(X);

% potentially select if we're reading from *.seg.nrrd
if isnumeric(segment_id)
    meta_names = fieldnames(meta);
    num_segments = sum(contains(meta_names, '_id'));
    if length(segment_id) == 1
        if ndims == 3 || size(X, 1) < num_segments
            layer_id = str2num(meta(1).(sprintf('segment%d_layer', segment_id-1)));
            label_value = str2num(meta(1).(sprintf('segment%d_labelvalue', segment_id-1)));
            if ndims == 3
                X = squeeze(uint8(X(:,:,:) == label_value));
            else
                X = squeeze(uint8(X(layer_id+1,:,:,:) == label_value));
            end
        else
            X = squeeze(X(segment_id,:,:,:));
        end
        X = permute(X, [2 1 3]);
    else
        Y = zeros(dims(length(dims)-2:length(dims)));
        for i=1:length(segment_id)
            curr_id = segment_id(i);
            if ndims == 3 || size(X, 1) < num_segments
                layer_id = str2num(meta(1).(sprintf('segment%d_layer', curr_id-1)));
                label_value = str2num(meta(1).(sprintf('segment%d_labelvalue', curr_id-1)));
                if ndims == 3
                    Y = or(Y, squeeze(uint8(X(:,:,:) == label_value)));
                else
                    Y = or(Y, squeeze(uint8(X(layer_id+1,:,:,:) == label_value)));
                end
            else
                size(squeeze(X(curr_id,:,:,:)))
                Y = or(Y, squeeze(X(curr_id,:,:,:)));
            end
        end
        X = permute(Y, [2 1 3]);
    end
end
end

% size(X)
% X = squeeze(X(segment_id,:,:,:));
% size(X)
% X = permute(X, [2 1 3]);



function datatype = getDatatype(metaType)

% Determine the datatype
switch (metaType)
 case {'signed char', 'int8', 'int8_t'}
  datatype = 'int8';
  
 case {'uchar', 'unsigned char', 'uint8', 'uint8_t'}
  datatype = 'uint8';

 case {'short', 'short int', 'signed short', 'signed short int', ...
       'int16', 'int16_t'}
  datatype = 'int16';
  
 case {'ushort', 'unsigned short', 'unsigned short int', 'uint16', ...
       'uint16_t'}
  datatype = 'uint16';
  
 case {'int', 'signed int', 'int32', 'int32_t'}
  datatype = 'int32';
  
 case {'uint', 'unsigned int', 'uint32', 'uint32_t'}
  datatype = 'uint32';
  
 case {'longlong', 'long long', 'long long int', 'signed long long', ...
       'signed long long int', 'int64', 'int64_t'}
  datatype = 'int64';
  
 case {'ulonglong', 'unsigned long long', 'unsigned long long int', ...
       'uint64', 'uint64_t'}
  datatype = 'uint64';
  
 case {'float'}
  datatype = 'single';
  
 case {'double'}
  datatype = 'double';
  
 otherwise
  assert(false, 'Unknown datatype')
end
end


function data = readData(fidIn, meta, datatype)

switch (meta.encoding)
 case {'raw'}
  
  data = fread(fidIn, inf, [datatype '=>' datatype]);
  
 case {'gzip', 'gz'}

  tmpBase = tempname;
  tmpFile = [tmpBase '.gz'];
  fidTmp = fopen(tmpFile, 'wb');
  assert(fidTmp > 3, 'Could not open temporary file for GZIP decompression')
  
  tmp = fread(fidIn, inf, 'uint8=>uint8');
  fwrite(fidTmp, tmp, 'uint8');
  fclose(fidTmp);
  
  gunzip(tmpFile)
  
  fidTmp = fopen(tmpBase, 'rb');
  
  meta.encoding = 'raw';
  data = readData(fidTmp, meta, datatype);
  
  fclose(fidTmp);
  cleaner = onCleanup(@()nrrdCleanup(tmpBase, tmpFile));
  
 case {'txt', 'text', 'ascii'}
  
  data = fscanf(fidIn, '%f');
  data = cast(data, datatype);
  
 otherwise
  assert(false, 'Unsupported encoding')
end

function nrrdCleanup(tmpBase, tmpFile)
delete(tmpBase);
delete(tmpFile);
end
end

function data = adjustEndian(data, meta)

[~,~,endian] = computer();

needToSwap = (isequal(endian, 'B') && isequal(lower(meta.endian), 'little')) || ...
             (isequal(endian, 'L') && isequal(lower(meta.endian), 'big'));
         
if (needToSwap)
    data = swapbytes(data);
end
end