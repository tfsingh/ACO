import { useState } from "react";

const useLocalStorageState = (key: any, defaultValue: any) => {
    const storedValue = localStorage.getItem(key);
    const initial = storedValue === '__empty__' ? '' : storedValue || defaultValue;

    const [state, setState] = useState(initial);

    const setLocalStorageState = (value: any) => {
        if (value === '__empty__') {
            localStorage.removeItem(key);
        } else {
            localStorage.setItem(key, value);
        }
        setState(value);
    };

    return [state, setLocalStorageState];
};

export default useLocalStorageState;