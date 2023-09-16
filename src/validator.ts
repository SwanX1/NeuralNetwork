export type ValidatableType =
  "string" | "number" | "boolean" | "object" | "array" | "function" | "symbol" | "bigint" | "undefined" | "null" | // Primitive types
  "number[]" | "number[][]" | "number[][][]" | "function"; // Complex types

export function validateValue<T>(
  value: unknown,
  type: ValidatableType,
): value is T {
  switch (type) {
    case "string": return typeof value === "string";
    case "number": return typeof value === "number";
    case "boolean": return typeof value === "boolean";
    case "object": return typeof value === "object" && value !== null;
    case "array": return Array.isArray(value);
    case "function": return typeof value === "function";
    case "symbol": return typeof value === "symbol";
    case "bigint": return typeof value === "bigint";
    case "undefined": return typeof value === "undefined";
    case "null": return value === null;
    case "number[]": return Array.isArray(value) && value.every(v => typeof v === "number");
    case "number[][]": return Array.isArray(value) && value.every(v => Array.isArray(v) && v.every(v => typeof v === "number"));
    case "number[][][]": return Array.isArray(value) && value.every(v => Array.isArray(v) && v.every(v => Array.isArray(v) && v.every(v => typeof v === "number")));
  }
}

export function validateObject<T extends Record<string, ValidatableType>>(
  obj: Record<string, unknown>,
  schema: T,
): obj is { [K in keyof T]: T[K] extends ValidatableType ? T[K] : never } {
  for (const key in schema) {
    if (!validateValue(obj[key], schema[key])) return false;
  }
  return true;
}

export function assertValue<T>(
  value: unknown,
  type: ValidatableType,
): asserts value is T {
  if (!validateValue<T>(value, type)) throw new Error(`Expected ${type}, got ${typeof value}`);
}

export function assertObject<T extends Record<string, ValidatableType>>(
  obj: Record<string, unknown>,
  schema: T,
): void | never {
  if (!validateObject(obj, schema)) {
    // typeof-ify the object
    const typedObj: { [K in keyof T]: T[K] extends ValidatableType ? T[K] : never } = {} as any;
    for (const key in schema) {
      typedObj[key] = obj[key] as any;
    }
  }
}